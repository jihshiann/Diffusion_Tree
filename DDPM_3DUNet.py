import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from typing import Optional
import re

# 定義常數，設定時間嵌入維度為 32，方便後續調整
TIME_EMB_DIM = 32

######################################
# 1. 數據前處理數據集（條件式）
######################################
class PeopleFlowDatasetCondition(Dataset):
    def __init__(self, csv_path: str, H: int, W: int, condition_length: int, prediction_length: int,
                 transform: Optional[callable] = None, normalize: bool = True):
        # 檢查 CSV 檔案是否存在，若不存在則拋出錯誤
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 檔案未找到，路徑為 {csv_path}")
        # 讀取 CSV 檔案到 DataFrame
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.condition_length = condition_length  # 條件序列長度（歷史數據）
        self.prediction_length = prediction_length  # 預測序列長度（未來數據）
        self.total_length = condition_length + prediction_length  # 總序列長度
        self.normalize = normalize  # 是否進行標準化
        
        # 篩選包含 "(lat, lon)" 的欄位，確保只有相關的流量數據
        flow_columns = [c for c in self.df.columns if '(' in c and ')' in c]
        if not flow_columns:
            raise ValueError("未找到有效的流量欄位，CSV 中無包含 '(lat, lon)' 的欄位。")
        if len(flow_columns) > H*W:
            flow_columns = flow_columns[:H*W]  # 若超過網格大小，截斷多餘欄位
        elif len(flow_columns) < H*W:
            raise ValueError(f"流量欄位數量 {len(flow_columns)} 小於網格大小 H*W = {H*W}。")
        
        # 解析欄位名稱中的緯度（lat）和經度（lon），並排序欄位
        def parse_lat_lon(column_name):
            # 使用正則表達式匹配 "(lat=X, lon=Y)" 格式，提取 X 和 Y
            match = re.search(r'\(lat=([\d.-]+), lon=([\d.-]+)\)', column_name)
            if match:
                lat = float(match.group(1))  # 提取緯度
                lon = float(match.group(2))  # 提取經度
                return lat, lon
            else:
                raise ValueError(f"欄位名稱格式無效：{column_name}")
        
        # 創建包含欄位名稱和其經緯度的列表
        column_info = [(col, *parse_lat_lon(col)) for col in flow_columns]
        # 根據緯度降序（北向南）、經度升序（西向東）排序
        sorted_column_info = sorted(column_info, key=lambda x: (-x[1], x[2]))
        sorted_flow_columns = [info[0] for info in sorted_column_info]
        
        # 驗證網格排列，輸出每個網格單元的欄位名稱和經緯度
        grid_arrangement = np.array(sorted_flow_columns).reshape(H, W)
        print("網格排列驗證：")
        for i in range(H):
            for j in range(W):
                col_name = grid_arrangement[i][j]
                lat, lon = parse_lat_lon(col_name)
                print(f"網格[{i},{j}]: {col_name}, lat={lat}, lon={lon}")
        
        # 使用排序後的欄位選擇數據
        flow_values = self.df[sorted_flow_columns].values  # 形狀：(N, num_points)
        num_points = flow_values.shape[1]
        
        if H * W != num_points:
            raise ValueError(f"網格大小 H*W = {H*W} 不匹配欄位數量 {num_points}。")
        
        # 重新塑形數據為 (N, H, W)，方便後續處理
        flow_2d = flow_values.reshape(-1, H, W).astype(np.float32)
        self.data = torch.from_numpy(flow_2d)  # 轉換為 PyTorch 張量，形狀：(N, H, W)
        
        if self.normalize:
            # 計算數據的均值和標準差，用於標準化
            self.mean_val = self.data.mean()
            self.std_val = self.data.std() + 1e-5  # 避免除零，添加小值
            if self.std_val < 1e-6:
                print("警告：標準差非常小，可能導致數值不穩定。")
            self.data = (self.data - self.mean_val) / self.std_val
        
        # 計算可用的序列數量，確保不超出數據範圍
        self.max_index = self.data.shape[0] - self.total_length + 1

    def __len__(self):
        # 返回可用的序列數量
        return self.max_index

    def __getitem__(self, idx):
        # 根據索引獲取條件序列（歷史數據）和目標序列（未來數據）
        cond_seq = self.data[idx : idx + self.condition_length]  # 形狀：(condition_length, H, W)
        target_seq = self.data[idx + self.condition_length : idx + self.total_length]  # 形狀：(prediction_length, H, W)
        # 添加通道維度，轉換為 (1, condition_length, H, W) 和 (1, prediction_length, H, W)
        cond_seq = cond_seq.unsqueeze(0)
        target_seq = target_seq.unsqueeze(0)
        if self.transform:
            # 若有轉換函數，應用於條件和目標序列
            cond_seq = self.transform(cond_seq)
            target_seq = self.transform(target_seq)
        return cond_seq, target_seq

def collate_fn(batch):
    # 批次處理函數，將多個樣本堆疊為批次
    conds, targets = zip(*batch)
    conds = torch.stack(conds, dim=0)  # 形狀：(B, 1, T_cond, H, W)
    targets = torch.stack(targets, dim=0)  # 形狀：(B, 1, T_pred, H, W)
    return conds, targets

######################################
# 2. 3D UNet 定義（接收條件時間嵌入）
######################################
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定義雙卷積層，包含兩層 3D 卷積、批次標準化和 ReLU 激活
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        # 前向傳播，通過雙卷積層處理輸入
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=TIME_EMB_DIM):
        super().__init__()
        # 初始化編碼器層 1，輸入通道數為 in_channels，輸出通道數為 base_channels
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        # 最大池化層，縮減空間維度，時間維度保持不變
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        # 初始化編碼器層 2，通道數加倍
        self.enc2 = DoubleConv3D(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))
        # 瓶頸層，通道數進一步加倍
        self.bottleneck = DoubleConv3D(base_channels*2, base_channels*4)
        # 上採樣層 2，恢復空間維度
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(1,3,3), stride=(1,2,2))
        # 解碼器層 2，結合跳躍連接
        self.dec2 = DoubleConv3D(base_channels*4, base_channels*2)
        # 上採樣層 1
        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,2,2), stride=(1,2,2))
        # 解碼器層 1
        self.dec1 = DoubleConv3D(base_channels*2, base_channels)
        # 輸出層，將通道數恢復為輸入通道數
        self.out_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)
        # 時間嵌入投影層，將時間嵌入投影到瓶頸層的通道數
        self.time_proj = nn.Linear(time_emb_dim, base_channels*4)

    def forward(self, x, t_emb):
        # 前向傳播，x 是輸入數據，t_emb 是時間和條件嵌入的組合
        e1 = self.enc1(x)  # 編碼器層 1 輸出
        p1 = self.pool1(e1)  # 池化層 1 輸出
        e2 = self.enc2(p1)  # 編碼器層 2 輸出
        p2 = self.pool2(e2)  # 池化層 2 輸出
        b = self.bottleneck(p2)  # 瓶頸層輸出
        # 將時間嵌入投影後加到瓶頸層，確保維度匹配
        t_emb_proj = self.time_proj(t_emb)
        b = b + t_emb_proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 廣播加法
        u2 = self.up2(b)  # 上採樣層 2
        cat2 = torch.cat([u2, e2], dim=1)  # 跳躍連接，結合編碼器層 2 輸出
        d2 = self.dec2(cat2)  # 解碼器層 2
        u1 = self.up1(d2)  # 上採樣層 1
        cat1 = torch.cat([u1, e1], dim=1)  # 跳躍連接，結合編碼器層 1 輸出
        d1 = self.dec1(cat1)  # 解碼器層 1
        out = self.out_conv(d1)  # 最終輸出層
        return out

######################################
# 3. DDPM 模型（條件式）
######################################
class DDPM3D(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__()
        self.model = model  # UNet3D 模型
        self.timesteps = timesteps  # 擴散步數
        self.device = device  # 設備（CPU 或 GPU）
        # 線性生成 beta 值，控制噪聲增加速度
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas  # 計算 alpha 值
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 累積 alpha 乘積
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # 平方根
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # 平方根
        # 條件投影層，將條件數據投影到時間嵌入維度
        self.cond_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),  # 全局平均池化，縮減到 1x1x1
            nn.Flatten(),  # 展平
            nn.Linear(1, TIME_EMB_DIM)  # 線性投影到 TIME_EMB_DIM
        ).to(device)

    def get_time_embedding(self, t):
        # 生成時間嵌入，使用正弦和餘弦函數，捕捉時間信息
        half_dim = TIME_EMB_DIM // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def get_condition_embedding(self, cond):
        # 獲取條件嵌入，通過投影層處理條件數據
        return self.cond_proj(cond)

    def q_sample(self, x0, t, noise=None):
        # 前向擴散過程，添加噪聲到原始數據 x0
        if noise is None:
            noise = torch.randn_like(x0)  # 若無噪聲，隨機生成
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1,1,1,1,1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1,1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise  # 根據 t 添加噪聲

    def p_losses(self, cond, x0, t):
        # 計算損失，比較模型預測的噪聲與真實噪聲
        noise = torch.randn_like(x0)  # 隨機生成噪聲
        x_t = self.q_sample(x0, t, noise=noise)  # 獲取噪聲數據
        time_emb = self.get_time_embedding(t).to(self.device)  # 獲取時間嵌入
        cond_emb = self.get_condition_embedding(cond)  # 獲取條件嵌入
        combined_emb = time_emb + cond_emb  # 結合時間和條件嵌入
        pred_noise = self.model(x_t, combined_emb)  # 模型預測噪聲
        return F.mse_loss(pred_noise, noise)  # 返回均方誤差損失

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        # 反向擴散過程，逐步去噪
        beta_t = self.betas[t].view(-1,1,1,1,1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t]).view(-1,1,1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1,1)
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        eps_theta = self.model(x_t, combined_emb)
        x_t_minus_1 = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)
        if (t > 0).any():
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            x_t_minus_1 += sigma_t * noise
        return x_t_minus_1

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        # 完整生成過程，從噪聲逐步生成數據
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x

######################################
# 4. 訓練與評估
######################################
def train_ddpm(model, diffusion, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda', patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for cond, target in train_loader:
            cond = cond.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
            loss = diffusion.p_losses(cond, target, t)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for cond, target in val_loader:
                cond = cond.to(device)
                target = target.to(device)
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                loss = diffusion.p_losses(cond, target, t)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training completed.")
    return model

@torch.no_grad()
def evaluate_reconstruction_mse(diffusion, dataset, device='cuda', max_samples=100):
    diffusion.eval()
    total_mse = 0
    count = 0
    N = min(len(dataset), max_samples)
    for i in range(N):
        cond, target = dataset[i]
        cond = cond.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        t_full = torch.full((1,), diffusion.timesteps - 1, device=device, dtype=torch.long)
        noise = torch.randn_like(target)
        x_T = diffusion.q_sample(target, t_full, noise=noise)
        x_recon = x_T.clone()
        for step in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((1,), step, device=device, dtype=torch.long)
            x_recon = diffusion.p_sample(x_recon, t_batch, cond)
        mse_val = F.mse_loss(x_recon, target).item()
        total_mse += mse_val
        count += 1
    return total_mse / count if count > 0 else 0.0

######################################
# 5. 主程式（時間序列分割）
######################################
if __name__ == "__main__":
    H = 22
    W = 22
    condition_length = 4
    prediction_length = 2
    batch_size = 4
    epochs = 10
    lr = 1e-4
    timesteps = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    full_dataset = PeopleFlowDatasetCondition(
        csv_path="all_merged.csv",
        H=H,
        W=W,
        condition_length = 4
    prediction_length = 2
    batch_size = 4
    epochs = 10
    lr = 1e-4
    timesteps = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    full_dataset = PeopleFlowDatasetCondition(
        csv_path="all_merged.csv",
        H=H,
        W=W,
        condition_length=condition_length,
        prediction_length=prediction_length,
        normalize=True
    )
    dataset_size = len(full_dataset)
    print("Full dataset length:", dataset_size)

    train_end = int(0.7 * dataset_size)
    val_end = int(0.85 * dataset_size)
    train_dataset = Subset(full_dataset, range(0, train_end))
    val_dataset = Subset(full_dataset, range(train_end, val_end))
    test_dataset = Subset(full_dataset, range(val_end, dataset_size))
    print(f"Dataset split: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    unet_3d = UNet3D(in_channels=1, base_channels=16, time_emb_dim=TIME_EMB_DIM)
    diffusion = DDPM3D(model=unet_3d, timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)

    trained_model = train_ddpm(unet_3d, diffusion, train_loader, val_loader, epochs=epochs, lr=lr, device=device, patience=3)

    sample_shape = (2, 1, prediction_length, H, W)
    cond_batch, _ = next(iter(val_loader))
    cond_batch = cond_batch.to(device)
    generated = diffusion.p_sample_loop(sample_shape, cond_batch[:2])
    print("Generated shape:", generated.shape)

    recon_mse = evaluate_reconstruction_mse(diffusion, val_dataset, device=device, max_samples=50)
    print(f"Reconstruction MSE (up to 50 samples): {recon_mse:.6f}")
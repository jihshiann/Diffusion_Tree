import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from collections import deque
import torch.optim as optim
from typing import Optional
import re
import logging

# 設定 logging 等級為 INFO，可根據需要調整為 DEBUG 以顯示更多細節
logging.basicConfig(level=logging.INFO)

# 定義時間嵌入的維度，後續模型中會用到此參數
TIME_EMB_DIM = 32

######################################
# 1. 資料前處理與數據集定義 (PeopleFlowDatasetCondition)
######################################
class PeopleFlowDatasetCondition(Dataset):
    """
    此類別負責從 CSV 檔案中讀取人流數據
    並根據指定網格大小 (H x W)、歷史序列長度 (condition_length) 與預測序列長度 (prediction_length)
    進行數據切割與預處理。
    """
    def __init__(self, csv_path: str, H: int, W: int, condition_length: int, prediction_length: int,
                 transform: Optional[callable] = None, normalize: bool = True, debug: bool = False):
        # 檢查 CSV 檔案是否存在，若不存在則拋出錯誤
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 檔案未找到，路徑為 {csv_path}")
        # 讀取 CSV 檔案
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        # 歷史與預測序列長度設定
        self.condition_length = condition_length
        self.prediction_length = prediction_length
        # 總序列長度 = 歷史長度 + 預測長度
        self.total_length = condition_length + prediction_length
        # 是否標準化數據
        self.normalize = normalize
        
        flow_columns = [c for c in self.df.columns if '(' in c and ')' in c]

        def parse_lat_lon(column_name):
            """
            解析欄位名稱中的經緯度。
    
            Args:
                column_name (str): 欄位名稱，格式如 "(lon,lat)"。
    
            Returns:
                tuple: (經度, 緯度)，分別為浮點數。
    
            Raises:
                ValueError: 若欄位名稱格式無效。
            """
            match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
            if match:
                lon = float(match.group(1))  # 提取經度
                lat = float(match.group(2))  # 提取緯度
                return lon, lat
            else:
                raise ValueError(f"欄位名稱格式無效：{column_name}")

        # 將每個欄位名稱解析為 (名稱, 經度, 緯度) 的元組列表
        column_info = [(col, *parse_lat_lon(col)) for col in flow_columns]

        # 將所有座標轉為 NumPy 陣列，方便計算，形狀為 (509, 2)
        coords = np.array([(lon, lat) for _, lon, lat in column_info])

        # 計算所有座標的平均經緯度，作為中心點的參考
        mean_lon = np.mean(coords[:, 0])  # 平均經度
        mean_lat = np.mean(coords[:, 1])  # 平均緯度

        # 計算每個座標到平均經緯度的歐氏距離，找到最接近的中心點
        distances_to_center = np.sqrt((coords[:, 0] - mean_lon)**2 + (coords[:, 1] - mean_lat)**2)
        central_idx = np.argmin(distances_to_center)  # 中心點的索引
        central_coord = coords[central_idx]  # 中心點的經緯度

        # 定義網格大小為 21x21，共 441 個位置
        grid_size = 21
        # 初始化網格，所有位置設為 -1，表示未分配
        grid = np.full((grid_size, grid_size), -1, dtype=int)
        # 設定中心點位置為 (10, 10)，並分配中心座標的索引
        central_row, central_col = 10, 10
        grid[central_row, central_col] = central_idx

        # 估計網格的經緯度間距（步長），基於座標差異的中位數
        lon_diffs = np.diff(np.sort(coords[:, 0]))  # 經度差異
        lat_diffs = np.diff(np.sort(coords[:, 1]))  # 緯度差異
        # 若差異存在，取中位數；否則使用預設值 0.005
        lon_step = np.median(lon_diffs[lon_diffs > 0]) if len(lon_diffs) > 0 else 0.005
        lat_step = np.median(lat_diffs[lat_diffs > 0]) if len(lat_diffs) > 0 else 0.005

        # 初始化可用座標索引列表，排除中心點
        available_indices = list(range(len(coords)))
        available_indices.remove(central_idx)

        # 生成按環層排序的網格位置列表，使用 Chebyshev 距離
        grid_positions = []
        for k in range(11):  # k 從 0 到 10，表示從中心到最外層
            for r in range(max(0, 10 - k), min(21, 10 + k + 1)):  # 行範圍
                for c in range(max(0, 10 - k), min(21, 10 + k + 1)):  # 列範圍
                    # 檢查是否屬於當前環層 k
                    if max(abs(r - 10), abs(c - 10)) == k:
                        grid_positions.append((r, c))

        # 按環層分配座標到網格
        for r, c in grid_positions:
            if grid[r, c] != -1:  # 若該位置已分配，跳過
                continue
    
            # 計算目標經緯度，基於中心點和網格間距
            target_lon = central_coord[0] + (c - central_col) * lon_step
            target_lat = central_coord[1] - (r - central_row) * lat_step
    
            # 設定方向約束，根據位置相對中心的關係
            lon_constraint = None
            if c < central_col:  # 左側，要求經度小於中心
                lon_constraint = lambda x: x < central_coord[0]
            elif c > central_col:  # 右側，要求經度大於中心
                lon_constraint = lambda x: x > central_coord[0]
    
            lat_constraint = None
            if r < central_row:  # 上方，要求緯度大於中心
                lat_constraint = lambda x: x > central_coord[1]
            elif r > central_row:  # 下方，要求緯度小於中心
                lat_constraint = lambda x: x < central_coord[1]
    
            # 篩選符合方向約束的座標索引
            filtered_indices = [idx for idx in available_indices if
                                (lon_constraint is None or lon_constraint(coords[idx][0])) and
                                (lat_constraint is None or lat_constraint(coords[idx][1]))]
    
            # 選擇最近的座標
            if filtered_indices:  # 若有符合約束的座標
                # 計算歐氏距離
                distances = np.sqrt((coords[filtered_indices, 0] - target_lon)**2 +
                                    (coords[filtered_indices, 1] - target_lat)**2)
                closest_idx = filtered_indices[np.argmin(distances)]  # 選最近的索引
            else:  # 若無符合約束的，放寬條件選最近的
                distances = np.sqrt((coords[available_indices, 0] - target_lon)**2 +
                                    (coords[available_indices, 1] - target_lat)**2)
                closest_idx = available_indices[np.argmin(distances)]
    
            # 分配座標到網格並更新可用索引
            grid[r, c] = closest_idx
            available_indices.remove(closest_idx)

        # 驗證網格是否填滿
        selected_indices = grid[grid != -1]
        if len(selected_indices) != grid_size * grid_size:
            raise ValueError(f"未能選取足夠的座標點，僅選取 {len(selected_indices)} 個，需 441 個。")

        # 將網格展平並生成排序後的 flow_columns 列表
        sorted_indices = grid.flatten()
        sorted_flow_columns = [column_info[idx][0] for idx in sorted_indices]

        import matplotlib.pyplot as plt

        def plot_grid(sorted_flow_columns, H, W):
            """
            顯示網格排列結果，確保經度 (x) 與緯度 (y) 順序正確，並避免圖片過大。
            """
            locations = [parse_lat_lon(col) for col in sorted_flow_columns]
            longitudes, latitudes = zip(*locations)  # 修正 x=經度, y=緯度

            plt.figure(figsize=(12, 12))  # 增加畫布大小
            plt.scatter(longitudes, latitudes, c='blue', marker='o', label='Grid Points')

            # 加上標籤
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    lon, lat = locations[idx]
                    plt.text(lon, lat, f'[{i},{j}]', fontsize=6, ha='right')  

            print(f"網格 [{i},{j}] 準備繪製標籤，經度 (x) = {lon}, 緯度 (y) = {lat}")

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("plot_grid")
            plt.grid(True)
            plt.legend()
            plt.savefig(r"C:\thesis\code\result_ddpm\plot_grid.png", dpi=600, bbox_inches='tight', pad_inches=0.1)

        # 呼叫函數顯示靜態圖
        plot_grid(sorted_flow_columns, H, W)

        # 轉換成 HxW 的網格
        grid_arrangement = np.array(sorted_flow_columns).reshape(H, W)
        
        # 從 DataFrame 中提取數值，並轉換成 numpy 陣列，形狀為 (N, H*W)
        flow_values = self.df[sorted_flow_columns].values
        num_points = flow_values.shape[1]
        if H * W != num_points:
            raise ValueError(f"網格大小 H*W = {H * W} 不匹配欄位數量 {num_points}。")
        
        # 重塑數據形狀成 (N, H, W)，方便後續轉換為 PyTorch 張量
        flow_2d = flow_values.reshape(-1, H, W).astype(np.float32)
        self.data = torch.from_numpy(flow_2d)
        
        # 若設定標準化，則計算全數據集的均值與標準差進行歸一化處理
        if self.normalize:
            self.mean_val = self.data.mean()
            self.std_val = self.data.std() + 1e-5  # 加入小值避免除零
            if self.std_val < 1e-6:
                logging.warning("警告：標準差非常小，可能導致數值不穩定。")
            self.data = (self.data - self.mean_val) / self.std_val
        
        # 計算可供切片的起始位置數量，確保不超出數據範圍
        self.max_index = self.data.shape[0] - self.total_length + 1

    def __len__(self):
        # 返回數據集中可用序列的數量
        return self.max_index

    def __getitem__(self, idx):
        # 根據索引取得條件序列與目標序列
        cond_seq = self.data[idx : idx + self.condition_length]
        target_seq = self.data[idx + self.condition_length : idx + self.total_length]
        # 為了與模型輸入匹配，增加 channel 維度 (例如從 [T, H, W] -> [1, T, H, W])
        cond_seq = cond_seq.unsqueeze(0)
        target_seq = target_seq.unsqueeze(0)
        # 若有額外轉換函數，則應用
        if self.transform:
            cond_seq = self.transform(cond_seq)
            target_seq = self.transform(target_seq)
        return cond_seq, target_seq

# 定義 collate_fn 用於 DataLoader 批次處理，將單個樣本疊加成 batch
def collate_fn(batch):
    conds, targets = zip(*batch)
    conds = torch.stack(conds, dim=0)   # 形狀: (B, 1, T_cond, H, W)
    targets = torch.stack(targets, dim=0)  # 形狀: (B, 1, T_pred, H, W)
    return conds, targets

######################################
# 2. 3D UNet 模型定義 (UNet3D 與 DoubleConv3D)
######################################
class DoubleConv3D(nn.Module):
    """
    定義 3D 卷積層，包含兩層卷積、批次正規化與 ReLU 激活函數
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    定義 3D UNet 模型，包含編碼器、解碼器、跳躍連接以及在瓶頸層中加入時間嵌入資訊
    """
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=TIME_EMB_DIM, dropout_rate=0.1):
        super().__init__()
        # 編碼器部分保持不變
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.bottleneck = DoubleConv3D(base_channels * 2, base_channels * 4)

        # 解碼器部分：調整上採樣層
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 
                                     kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        
        # 在 self.up1 中添加 output_padding=(0,1,1)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 
                                     kernel_size=(1,2,2), stride=(1,2,2), 
                                     output_padding=(0,1,1))
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        
        # 輸出層保持不變
        self.out_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, base_channels * 4),
            nn.SiLU()
        )

    def forward(self, x, t_emb):
        # 編碼過程
        e1 = self.enc1(x)     # (B, base_channels, T, 21, 21)
        p1 = self.pool1(e1)   # (B, base_channels, T, 10, 10)
        e2 = self.enc2(p1)    # (B, base_channels*2, T, 10, 10)
        p2 = self.pool2(e2)   # (B, base_channels*2, T, 5, 5)
        b = self.bottleneck(p2)  # (B, base_channels*4, T, 5, 5)

        b = self.dropout(b)
        t_emb_proj = self.time_proj(t_emb)
        b = b + t_emb_proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 解碼過程
        u2 = self.up2(b)      # (B, base_channels*2, T, 10, 10)
        cat2 = torch.cat([u2, e2], dim=1)  # (B, base_channels*4, T, 10, 10)
        d2 = self.dec2(cat2)  # (B, base_channels*2, T, 10, 10)
        u1 = self.up1(d2)     # (B, base_channels, T, 21, 21) 因 output_padding
        cat1 = torch.cat([u1, e1], dim=1)  # (B, base_channels*2, T, 21, 21)
        d1 = self.dec1(cat1)  # (B, base_channels, T, 21, 21)
        out = self.out_conv(d1)  # (B, in_channels, T, 21, 21)
        return out

######################################
# 3. DDPM 模型定義 (DDPM3D)
######################################
class DDPM3D(nn.Module):
    """
    定義條件式 DDPM 模型，結合 3D UNet 與時間、條件嵌入，包含正向擴散 (q_sample)
    與反向去噪 (p_sample, p_sample_loop) 過程，以及計算損失 (p_losses)
    """
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__()
        self.model = model      # 傳入的 UNet3D 模型
        self.timesteps = timesteps
        self.device = device
        # 線性生成 beta 值，用來控制每一步添加噪聲的程度
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        # 計算 alpha 值 (1 - beta)
        self.alphas = 1.0 - self.betas
        # alpha 累積乘積，用於直接計算某一步的噪聲影響
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 各步驟 alpha 累積乘積的平方根
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # 1 - alpha 累積乘積的平方根
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # 條件投影層：將條件數據 (通常來自歷史序列) 經全局平均池化後映射到 TIME_EMB_DIM 維度
        self.cond_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(1, TIME_EMB_DIM)
        ).to(device)
        # 預先計算時間嵌入中使用的頻率因子，避免在每次呼叫 get_time_embedding 時重複計算
        self.half_dim = TIME_EMB_DIM // 2
        self.freq_factor = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) *
                                     -(math.log(10000.0) / (self.half_dim - 1))).to(device)

    def get_time_embedding(self, t):
        """
        生成時間嵌入：使用正弦與餘弦函數捕捉時間特徵
        t 為時間步驟張量，形狀為 (B,)。
        """
        t = t.float()
        # 生成正弦與餘弦嵌入，形狀變換為 (B, half_dim)
        emb = t[:, None] * self.freq_factor.to(t.device)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def get_condition_embedding(self, cond):
        """
        將條件數據透過條件投影層轉換為 TIME_EMB_DIM 維度的向量
        """
        return self.cond_proj(cond)

    def q_sample(self, x0, t, noise=None):
        """
        正向擴散過程：根據時間步驟 t 將原始數據 x0 添加噪聲
        若未提供 noise，則使用標準正態分布隨機生成
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # 根據 t 取得對應的 sqrt(alpha_cumprod) 與 sqrt(1 - alpha_cumprod)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def p_losses(self, cond, x0, t):
        """
        損失計算函數：先生成含噪數據，再由模型預測噪聲，最後以均方誤差 (MSE) 與真實噪聲比較
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        # 生成時間嵌入與條件嵌入，並將兩者相加
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        # 模型預測噪聲
        pred_noise = self.model(x_t, combined_emb)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        """
        單步反向去噪：根據當前數據 x_t 及時間步 t，利用模型預測噪聲，
        並計算 x_(t-1) 的估計值，若 t > 0 則額外添加隨機噪聲。
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        # 生成時間與條件嵌入
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        # 模型預測噪聲 eps_theta
        eps_theta = self.model(x_t, combined_emb)
        # 計算 x_(t-1) 的估計值 (無噪聲版本)
        x_t_minus_1 = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)
        # 根據每個樣本的 t 值決定是否加上隨機噪聲 (t > 0 時才添加)
        mask = (t > 0).float().view(-1, 1, 1, 1, 1)
        sigma_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        x_t_minus_1 = x_t_minus_1 + mask * sigma_t * noise
        return x_t_minus_1

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        完整反向生成過程：
        從純隨機噪聲開始，逐步呼叫 p_sample 去噪直到 t = 0，返回最終生成結果
        """
        x = torch.randn(shape, device=self.device)
        # 從最後一步 (最大 t) 反向迭代至 0
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x

######################################
# 4. 訓練與評估函數
######################################
def train_ddpm(diffusion, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda', patience=3, 
               lr_scheduler=True, weight_decay=1e-6):
    """
    訓練 DDPM 模型：
    - diffusion：DDPM3D 模型
    - train_loader：訓練數據 DataLoader
    - val_loader：驗證數據 DataLoader
    - epochs：訓練週期數
    - lr：學習率
    - patience：早停機制耐心值
    """
    # 使用整個 diffusion 模型 (包括 UNet3D 與條件投影層) 的參數進行更新
    # 使用 weight_decay 減少過擬合
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    diffusion.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        diffusion.train()
        total_train_loss = 0
        # 逐批次訓練
        for cond, target in train_loader:
            cond = cond.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # 隨機抽取一個時間步 t 作為當前步驟
            t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
            loss = diffusion.p_losses(cond, target, t)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        # 驗證階段：不更新參數，只計算驗證損失
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for cond, target in val_loader:
                cond = cond.to(device)
                target = target.to(device)
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                loss = diffusion.p_losses(cond, target, t)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {avg_val_loss:.4f}")

        # 根據驗證損失進行模型保存或早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(diffusion.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("Training completed.")
    return diffusion

@torch.no_grad()
def evaluate_model(diffusion, dataset, device='cuda', max_samples=100):
    """
    綜合評估模型性能：包括 MSE、MAE、SSIM 等多種指標
    """
    diffusion.eval()
    metrics = {
        'mse': 0.0,
        'mae': 0.0,
        # 可添加其他指標
    }
    count = 0
    N = min(len(dataset), max_samples)
    
    for i in range(N):
        cond, target = dataset[i]
        cond = cond.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # 生成預測
        x_recon = diffusion.p_sample_loop(target.shape, cond)
        
        # 計算多種評估指標
        metrics['mse'] += F.mse_loss(x_recon, target).item()
        metrics['mae'] += F.l1_loss(x_recon, target).item()
        # 可添加更多指標計算
        
        count += 1
    
    # 計算平均值
    for key in metrics:
        metrics[key] /= count if count > 0 else 1.0
    
    return metrics

######################################
# 5. 主程式：數據分割與模型訓練、生成、評估
######################################
if __name__ == "__main__":
    # 網格尺寸與序列長度設定
    H = 21                     # 網格高度
    W = 21                     # 網格寬度
    condition_length = 4       # 歷史條件序列長度 (小時數)
    prediction_length = 2      # 預測序列長度 (小時數)
    batch_size = 4             # 每個 batch 的樣本數
    epochs = 10                # 訓練週期數
    lr = 1e-4                  # 學習率
    timesteps = 200            # 擴散步數

    # 設定種子以確保實驗可重複性
    torch.manual_seed(42)
    np.random.seed(42)

    # 根據是否有 GPU 決定設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化數據集，指定 CSV 檔案路徑 (請確認檔案名稱與路徑正確)
    full_dataset = PeopleFlowDatasetCondition(
        csv_path=r"C:\thesis\code\Taipei_CF\all_merged.csv",
        H=H,
        W=W,
        condition_length=condition_length,
        prediction_length=prediction_length,
        normalize=True,
        debug=True
    )
    dataset_size = len(full_dataset)
    logging.info(f"Full dataset length: {dataset_size}")

    # 按比例分割數據集：70% 訓練、15% 驗證、15% 測試
    train_end = int(0.7 * dataset_size)
    val_end = int(0.85 * dataset_size)
    train_dataset = Subset(full_dataset, range(0, train_end))
    val_dataset = Subset(full_dataset, range(train_end, val_end))
    test_dataset = Subset(full_dataset, range(val_end, dataset_size))
    logging.info(f"Dataset split: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")

    # 建立 DataLoader，用於批次讀取數據
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化 3D UNet 模型，並建立 DDPM 模型
    unet_3d = UNet3D(in_channels=1, base_channels=16, time_emb_dim=TIME_EMB_DIM)
    diffusion = DDPM3D(model=unet_3d, timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)

    # 開始訓練 DDPM 模型
    trained_diffusion = train_ddpm(diffusion, train_loader, val_loader, epochs=epochs, lr=lr, device=device, patience=3)

    # 測試生成流程：根據部分條件數據生成預測結果
    sample_shape = (2, 1, prediction_length, H, W)   # 生成 2 個樣本，每個樣本形狀為 (1, prediction_length, H, W)
    cond_batch, _ = next(iter(val_loader))
    cond_batch = cond_batch.to(device)
    generated = trained_diffusion.p_sample_loop(sample_shape, cond_batch[:2])
    logging.info(f"Generated shape: {generated.shape}")

    # 評估模型重構誤差 (MSE)，使用部分驗證數據進行測試
    recon_mse = evaluate_model(trained_diffusion, val_dataset, device=device, max_samples=50)
    logging.info(f"Reconstruction MSE (up to 50 samples): {recon_mse['mse']:.6f}")


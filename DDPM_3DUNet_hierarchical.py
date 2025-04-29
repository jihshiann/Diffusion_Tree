import os
import re
import math
import json
import logging
import random
import numpy as np
import pandas as pd
import scipy.linalg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from typing import Optional
from torchvision.models import inception_v3, Inception_V3_Weights

logging.basicConfig(level=logging.INFO)
TIME_EMB_DIM = 128

def parse_lat_lon(column_name: str) -> tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class PeopleFlowDatasetCondition(Dataset):
    def __init__(self, csv_path: str, H: int, W: int, condition_length: int, 
                 prediction_length: int, transform: Optional[callable] = None, 
                 normalize: bool = True, debug: bool = False):
        """
        初始化網格數據集，同時讀取非流量（額外條件）數據，並處理風向等特徵。
        對額外條件的處理方式：
          1. 對於 '最大陣風風向' 與 '風向'，計算 sin 與 cos 後捨棄原始數值。
          2. 對 categorical 特徵 "holiday" 與 "月" 進行 one-hot 編碼，
             其他連續特徵如啟用 normalize 則正規化，並保存正規化參數。
        Args:
            csv_path (str): CSV 文件路徑
            H (int): 網格高度
            W (int): 網格寬度
            condition_length (int): 條件數據長度
            prediction_length (int): 預測數據長度
            transform (callable, optional): 數據轉換函數
            normalize (bool): 是否標準化數據
            debug (bool): 是否啟用調試模式
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 檔案未找到：{csv_path}")
        # 讀取 CSV 並刪除第一個欄位 (時間)
        self.df = pd.read_csv(csv_path)
        self.df = self.df.iloc[:, 1:]
        
        self.transform = transform
        self.condition_length = condition_length
        self.prediction_length = prediction_length
        self.total_length = condition_length + prediction_length
        self.normalize = normalize
        self.H, self.W = H, W

        # ---------------------------
        # 1. 針對流量資料：選取欄位含有括號(經緯度) 的欄位
        flow_columns = [c for c in self.df.columns if '(' in c and ')' in c]
        column_info = [(col, *parse_lat_lon(col)) for col in flow_columns]
        coords = np.array([(lon, lat) for _, lon, lat in column_info])

        # 計算中心點
        mean_lon, mean_lat = np.mean(coords, axis=0)
        distances_to_center = np.sqrt((coords[:, 0] - mean_lon)**2 + (coords[:, 1] - mean_lat)**2)
        central_idx = np.argmin(distances_to_center)
        central_coord = coords[central_idx]

        # 初始化網格
        grid = np.full((H, W), -1, dtype=int)
        central_row, central_col = H // 2, W // 2
        grid[central_row, central_col] = central_idx

        # 計算經緯度步長
        lon_diffs = np.diff(np.sort(coords[:, 0]))
        lat_diffs = np.diff(np.sort(coords[:, 1]))
        lon_step = np.median(lon_diffs[lon_diffs > 0]) if len(lon_diffs) > 0 else 0.005
        lat_step = np.median(lat_diffs[lat_diffs > 0]) if len(lat_diffs) > 0 else 0.005

        # 建立可用索引與網格位置（以曼哈頓距離層級）
        available_indices = list(range(len(coords)))
        available_indices.remove(central_idx)
        grid_positions = []
        max_dist = max(H // 2, W // 2)
        for k in range(max_dist + 1):
            for r in range(max(0, central_row - k), min(H, central_row + k + 1)):
                for c in range(max(0, central_col - k), min(W, central_col + k + 1)):
                    if max(abs(r - central_row), abs(c - central_col)) == k:
                        grid_positions.append((r, c))

        # 優化座標分配
        for r, c in grid_positions:
            if grid[r, c] != -1:
                continue
            target_lon = central_coord[0] + (c - central_col) * lon_step
            target_lat = central_coord[1] - (r - central_row) * lat_step

            # 設定經緯度約束
            lon_constraint = None
            if c < central_col:
                lon_constraint = lambda x: x < central_coord[0]
            elif c > central_col:
                lon_constraint = lambda x: x > central_coord[0]
            lat_constraint = None
            if r < central_row:
                lat_constraint = lambda x: x > central_coord[1]
            elif r > central_row:
                lat_constraint = lambda x: x < central_coord[1]

            filtered_indices = [idx for idx in available_indices if
                                (lon_constraint is None or lon_constraint(coords[idx][0])) and
                                (lat_constraint is None or lat_constraint(coords[idx][1]))]

            if filtered_indices:
                distances = np.sqrt((coords[filtered_indices, 0] - target_lon)**2 +
                                    (coords[filtered_indices, 1] - target_lat)**2)
                closest_idx = filtered_indices[np.argmin(distances)]
            else:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                neighbor_coords = []
                for nr, nc in neighbors:
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != -1:
                        neighbor_coords.append(coords[grid[nr, nc]])
                if neighbor_coords:
                    neighbor_mean = np.mean(neighbor_coords, axis=0)
                    distances = np.sqrt((coords[available_indices, 0] - neighbor_mean[0])**2 +
                                        (coords[available_indices, 1] - neighbor_mean[1])**2)
                    closest_idx = available_indices[np.argmin(distances)]
                else:
                    distances = np.sqrt((coords[available_indices, 0] - target_lon)**2 +
                                        (coords[available_indices, 1] - target_lat)**2)
                    closest_idx = available_indices[np.argmin(distances)]

            grid[r, c] = closest_idx
            available_indices.remove(closest_idx)

        if len(grid[grid != -1]) != H * W:
            raise ValueError(f"網格未填滿：選取 {len(grid[grid != -1])} 個，需 {H * W} 個")

        sorted_indices = grid.flatten()
        self.sorted_flow_columns = [column_info[idx][0] for idx in sorted_indices]
        self._plot_grid(save_path=r"C:\thesis\code\result_ddpm_hierarchical\plot_grid.png") 
        self._plot_grid_matrix(save_path=r"C:\thesis\code\result_ddpm_hierarchical\plot_grid_matrix.png") 

        flow_values = self.df[self.sorted_flow_columns].values.reshape(-1, H, W).astype(np.float32)
        self.data = torch.from_numpy(flow_values)

        if normalize:
            self.mean_val = self.data.mean()
            self.std_val = self.data.std() + 1e-5
            self.data = (self.data - self.mean_val) / self.std_val

        # ---------------------------
        # 2. 處理額外條件欄位 (非 flow 部分)
        # 針對 '最大陣風風向' 與 '風向'，計算 sin 與 cos 後刪除原始欄位
        for col in ['最大陣風風向', '風向']:
            if col in self.df.columns:
                self.df[f'sin_{col}'] = np.sin(np.deg2rad(self.df[col]))
                self.df[f'cos_{col}'] = np.cos(np.deg2rad(self.df[col]))
                self.df.drop(columns=[col], inplace=True)
        extra_cols_list = [
            "測站氣壓", "海平面氣壓", "氣溫", "露點溫度", "相對溼度", "風速", "最大陣風",
            "降水量", "降水時數", "日照時數", "全天空日射量", "能見度", "紫外線指數", "總雲量",
            "holiday", "weekday", "年", "月", "日", "時",
            "sin_風向", "cos_風向", "sin_最大陣風風向", "cos_最大陣風風向"
        ]
        if "hoilday" in self.df.columns:
            self.df.rename(columns={"hoilday": "holiday"}, inplace=True)
        # 取出額外條件資料
        df_extra = self.df[extra_cols_list].copy()
        # 定義 categorical 欄位，這裡要求 "holiday" 與 "月" 進行 one-hot 編碼
        cat_features = ['holiday', '月']
        df_extra[cat_features] = df_extra[cat_features].astype(str)
        df_cat = pd.get_dummies(df_extra[cat_features], prefix=cat_features)
        # 連續特徵部分：將 categorical 欄位剔除
        df_cont = df_extra.drop(columns=cat_features)
        if self.normalize:
            cont_mean = df_cont.mean()
            cont_std = df_cont.std() + 1e-5
            df_cont = (df_cont - cont_mean) / cont_std
            # 保存額外條件連續特徵的正規化參數，便於日後保持一致
            self.extra_cont_mean = cont_mean
            self.extra_cont_std = cont_std
        # 將正規化後的連續特徵和 one-hot 的 categorical 特徵合併
        df_extra_processed = pd.concat([df_cont, df_cat], axis=1)
        self.extra_columns = list(df_extra_processed.columns)
        print("Extra Columns:", df_extra_processed.columns.tolist())
        self.extra_data = df_extra_processed.values.astype(np.float32)

        self.max_index = self.data.shape[0] - self.total_length + 1

    def _plot_grid(self, save_path: str):
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        locations = [parse_lat_lon(col) for col in self.sorted_flow_columns]
        longitudes, latitudes = zip(*locations)
        plt.figure(figsize=(12, 12))
        plt.scatter(longitudes, latitudes, c='blue', marker='o', label='Grid Points')
        for i in range(self.H):
            for j in range(self.W):
                idx = i * self.W + j
                plt.text(longitudes[idx], latitudes[idx], f'[{i},{j}]', fontsize=6, ha='right')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Grid Arrangement")
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _plot_grid_matrix(self, save_path: str):
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig, ax = plt.subplots(figsize=(21, 21))
        ax.set_xticks(np.arange(self.W))
        ax.set_yticks(np.arange(self.H))
        ax.set_xticklabels(np.arange(self.W))
        ax.set_yticklabels(np.arange(self.H))
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.set_title("Grid Matrix with Coordinates")
        for i in range(self.H):
            for j in range(self.W):
                idx = i * self.W + j
                coord = parse_lat_lon(self.sorted_flow_columns[idx])
                lon, lat = coord[0], coord[1]
                text = f'{lon:.3f}\n{lat:.3f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=10)
        ax.grid(True, which='both', linestyle='-', linewidth=1)
        plt.gca().invert_yaxis()
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def __len__(self) -> int:
        return self.max_index

    def __getitem__(self, idx):
        # 取出 flow 資料序列
        cond_seq = self.data[idx:idx + self.condition_length]  
        target_seq = self.data[idx + self.condition_length:idx + self.total_length]  
        model_input = torch.cat([cond_seq, target_seq], dim=0).unsqueeze(0)
        # 取出額外條件資料，形狀為 [1, total_length, num_extra_features]
        extra_cond_seq = torch.from_numpy(self.extra_data[idx:idx + self.total_length])
        extra_cond_seq = extra_cond_seq.unsqueeze(0)
        return model_input, target_seq.unsqueeze(0), extra_cond_seq

def collate_fn(batch):
    conds, targets, extra_conds = zip(*batch)
    return torch.stack(conds), torch.stack(targets), torch.stack(extra_conds)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
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
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, dropout_rate=0.0):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=(2, 2, 2), stride=(1, 2, 2), output_padding=(0, 1, 1))
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2), output_padding=(1, 0, 0))
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.time_proj = nn.Sequential(nn.Linear(time_emb_dim, base_channels * 8), nn.SiLU())
        self.x_full_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
    def forward(self, x_t, x_full, t_emb):
        x_full_conv = self.x_full_conv(x_full)
        x_input = x_t + x_full_conv
        e1 = self.enc1(x_input)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        p4 = self.pool4(e4)
        t_emb = self.time_proj(t_emb)[:, :, None, None, None]
        b = self.bottleneck(p4 + t_emb)
        b = self.dropout(b)
        d4 = self.up4(b)
        if d4.shape[-3:] != e4.shape[-3:]:
            d4 = F.interpolate(d4, size=e4.shape[-3:], mode='trilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        if d3.shape[-3:] != e3.shape[-3:]:
            d3 = F.interpolate(d3, size=e3.shape[-3:], mode='trilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = F.interpolate(d2, size=e2.shape[-3:], mode='trilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = F.interpolate(d1, size=e1.shape[-3:], mode='trilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out_conv(d1)
        return out[:, :, :1, :, :]

class DDPM3D(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, 
                 beta_start: float = 1e-4, beta_end: float = 0.02, 
                 device: str = 'cuda', condition_dim: int = 5):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.half_dim = TIME_EMB_DIM // 2
        self.freq_factor = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) *
                                     -(math.log(10000.0) / (self.half_dim - 1))).to(device)
        # 新增條件嵌入 MLP；condition_dim 為額外條件特徵數
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, TIME_EMB_DIM)
        )
    def get_time_embedding(self, t):
        t = t.float()
        emb = t[:, None] * self.freq_factor.to(t.device)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    def get_condition_embedding(self, extra_cond):
        # 此處僅取第一個時間步的額外條件進行映射，並與時間嵌入直接相加融合
        # 之後考慮拼接後再經過額外全連接層
        # 之後考慮所有時間步都接收額外條件
        # extra_cond 形狀：[batch_size, 1, total_length, condition_dim] = [150, 1, 9, 35]
        # 選取第一個時間步
        if extra_cond.dim() == 4:
            # 訓練時的形狀：[batch_size, 1, total_length, condition_dim]
            extra_cond = extra_cond[:, 0, 0, :]  # [batch_size, condition_dim]
        elif extra_cond.dim() == 3:
            # 評估時的形狀：[batch_size, total_length, condition_dim]
            extra_cond = extra_cond[:, 0, :]  # [batch_size, condition_dim]
        else:
            raise ValueError(f"Unexpected extra_cond shape: {extra_cond.shape}")
        cond_emb = self.condition_mlp(extra_cond)  # [batch_size, TIME_EMB_DIM]
        return cond_emb
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    def p_losses(self, cond, target, extra_cond, t):
        x_full = torch.cat([target, cond[:, :, 1:]], dim=2)
        noise = torch.randn_like(target)
        x_noisy_target = self.q_sample(target, t, noise=noise)
        x_t = torch.cat([x_noisy_target, cond[:, :, 1:]], dim=2)
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(extra_cond)
        combined_emb = time_emb + cond_emb
        pred_noise = self.model(x_t, x_full, combined_emb)
        pred_noise_target = pred_noise[:, :, :1, :, :]
        return F.mse_loss(pred_noise_target, noise)
    @torch.no_grad()
    def p_sample(self, x_t, t, cond, extra_cond):
        if x_t.dim() == 4:
            x_t = x_t.unsqueeze(1)
        if cond.dim() == 4:
            cond = cond.unsqueeze(1)
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(extra_cond)
        combined_emb = time_emb + cond_emb
        x_t_full = torch.cat([x_t, cond[:, :, 1:]], dim=2)
        x_full = cond
        eps_theta = self.model(x_t_full, x_full, combined_emb)
        eps_theta_target = eps_theta[:, :, :1, :, :]
        x_t_minus_1 = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta_target)
        mask = (t > 0).float().view(-1, 1, 1, 1, 1)
        sigma_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        return x_t_minus_1 + mask * sigma_t * noise
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, extra_cond):
        if cond.dim() == 4:
            cond = cond.unsqueeze(1)
        if len(shape) == 4:
            batch_size = shape[0]
            shape = (batch_size, 1, shape[1], shape[2], shape[3])
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond, extra_cond)
        return x

# --------------------------------------
# 視覺化工具：用於繪製預測結果、誤差網格圖等
# --------------------------------------
def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """
    截斷 colormap，僅使用其中一部分的色階範圍。
    參數:
        cmap: 原始的 colormap
        minval, maxval: 取色範圍
        n: 取樣點數
    回傳:
        新的截斷後的 colormap
    """
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def visualize_predictions(cond, generated, target, sample_idx: int = 0, 
                         save_dir: str = r"C:\thesis\code\result_ddpm_hierarchical"):
    """
    視覺化預測結果與真實值的比較，包含生成結果、真實數據、以及誤差（MSE 與 MAE）的圖形。
    參數:
        cond: 條件數據
        generated: 生成結果
        target: 真實目標數據
        sample_idx: 指定要視覺化哪個樣本
        save_dir: 圖形存檔的目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    pred_length = generated.shape[2]
    
    # 若 sample_idx 為 None，則計算所有樣本的平均值
    if sample_idx is None:
        generated_avg = torch.mean(generated, dim=(0, 2)).squeeze(0).cpu().numpy()  # (H, W)
        target_avg = torch.mean(target, dim=(0, 2)).squeeze(0).cpu().numpy()        # (H, W)
        
        mse_matrix = (generated_avg - target_avg) ** 2
        mae_matrix = np.abs(generated_avg - target_avg)
        mape_matrix = np.abs((target_avg - generated_avg) / (target_avg + 1e-10)) * 100
        smape_matrix = np.abs(generated_avg - target_avg) / (np.abs(target_avg) + np.abs(generated_avg) + 1e-10) * 100
        
        mse = np.mean(mse_matrix)
        mae = np.mean(mae_matrix)
        mape = np.mean(mape_matrix)
        smape = np.mean(smape_matrix)
        
        # 子圖 1：平均生成結果
        plt.figure(figsize=(6, 6))
        plt.imshow(generated_avg, cmap='viridis')
        plt.colorbar()
        plt.title('avg_generated')
        plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_generated.png'), dpi=300)
        plt.close()
        
        # 子圖 2：平均真實值
        plt.figure(figsize=(6, 6))
        plt.imshow(target_avg, cmap='viridis')
        plt.colorbar()
        plt.title('avg_target')
        plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_target.png'), dpi=300)
        plt.close()
        
        # 子圖 3：MSE 誤差圖
        plt.figure(figsize=(6, 6))
        plt.imshow(mse_matrix, cmap='hot')
        plt.colorbar()
        plt.title(f'MSE: {mse:.0f}')
        plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_mse.png'), dpi=300)
        plt.close()
        
        # 子圖 4：MAE 誤差圖（標示整數）
        plt.figure(figsize=(6, 6))
        plt.imshow(mae_matrix, cmap='hot')
        plt.colorbar()
        for i in range(mae_matrix.shape[0]):
            for j in range(mae_matrix.shape[1]):
                plt.text(j, i, f'{int(round(mae_matrix[i, j]))}', ha='center', va='center', color='white', fontsize=4)
        plt.title(f'MAE: {mae:.0f}')
        plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_mae.png'), dpi=300)
        plt.close()
        
        # 子圖 5：MAPE 誤差圖（標示整數）
        plt.figure(figsize=(6, 6))
        plt.imshow(mape_matrix, cmap='hot')
        plt.colorbar()
        for i in range(mape_matrix.shape[0]):
            for j in range(mape_matrix.shape[1]):
                plt.text(j, i, f'{int(round(mape_matrix[i, j]))}', ha='center', va='center', color='white', fontsize=4)
        plt.title(f'MAPE: {mape:.0f}%')
        plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_mape.png'), dpi=300)
        plt.close()
        
        # 子圖 6：SMAPE 誤差圖（標示整數）
        plt.figure(figsize=(6, 6))
        plt.imshow(smape_matrix, cmap='hot')
        plt.colorbar()
        for i in range(smape_matrix.shape[0]):
            for j in range(smape_matrix.shape[1]):
                plt.text(j, i, f'{int(round(smape_matrix[i, j]))}', ha='center', va='center', color='white', fontsize=4)
        plt.title(f'SMAPE: {smape:.0f}%')
        plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_smape.png'), dpi=300)
        plt.close()
    
    else:
        # 原有單樣本視覺化邏輯（這裡保留，僅更新為包含 MAPE 和 SMAPE）
        for t in range(pred_length):
            plt.figure(figsize=(20, 4))
            
            plt.subplot(1, 5, 1)
            plt.imshow(generated[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title(f'Generated (t={t})')
            
            plt.subplot(1, 5, 2)
            plt.imshow(target[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title(f'True (t={t})')
            
            error_sq = (generated[sample_idx, 0, t].cpu().numpy() - target[sample_idx, 0, t].cpu().numpy()) ** 2
            plt.subplot(1, 5, 3)
            plt.imshow(error_sq, cmap='hot')
            plt.colorbar()
            plt.title(f'MSE (t={t})')
            
            error_abs = np.abs(generated[sample_idx, 0, t].cpu().numpy() - target[sample_idx, 0, t].cpu().numpy())
            plt.subplot(1, 5, 4)
            plt.imshow(error_abs, cmap='hot')
            plt.colorbar()
            plt.title(f'MAE (t={t})')
            
            # 新增 MAPE 子圖
            mape = np.abs((target[sample_idx, 0, t].cpu().numpy() - generated[sample_idx, 0, t].cpu().numpy()) / 
                         (target[sample_idx, 0, t].cpu().numpy() + 1e-10)) * 100
            plt.subplot(1, 5, 5)
            plt.imshow(mape, cmap='hot')
            plt.colorbar()
            plt.title(f'MAPE (t={t})')
            
            plt.suptitle(f'Sample {sample_idx} - Time Step {t}')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(save_dir, f'prediction_sample{sample_idx}_t{t}.png'), dpi=300)
            plt.close()

def plot_grid_with_error(sorted_flow_columns: list, H: int, W: int, 
                         mse_matrix: np.ndarray, mae_matrix: np.ndarray, mape_matrix: np.ndarray, 
                         save_dir: str = r"C:\\thesis\\code\\result_ddpm_hierarchical", smape_matrix: np.ndarray = None):
    """
    繪製網格圖，顯示每個網格點的誤差（MSE、MAE 和 MAPE），並將結果存成圖與表格。
    
    Args:
        sorted_flow_columns (list): 經緯度欄位的排序列表。
        H (int): 網格高度。
        W (int): 網格寬度。
        mse_matrix (np.ndarray): 每個網格點的 MSE 矩陣，形狀為 (H, W)。
        mae_matrix (np.ndarray): 每個網格點的 MAE 矩陣，形狀為 (H, W)。
        mape_matrix (np.ndarray): 每個網格點的 MAPE 矩陣，形狀為 (H, W)。
        save_dir (str): 存檔路徑。
        smape_matrix (np.ndarray, optional): 每個網格點的 SMAPE 矩陣，形狀為 (H, W)。預設為 None。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    locations = [parse_lat_lon(col) for col in sorted_flow_columns]
    longitudes, latitudes = zip(*locations)
    
    orig_cmap = plt.get_cmap('OrRd')
    trunc_cmap = truncate_colormap(orig_cmap, 0.3, 1.0)
    
    # 繪製 MSE 網格圖
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mse_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MSE')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MSE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mse.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 繪製 MAE 網格圖
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mae_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAE')
    for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
        plt.text(lon, lat, f'{int(round(mae_matrix.flatten()[i]))}', ha='center', va='center', color='black', fontsize=5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mae.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mae_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAE')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mae_clean.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 繪製 MAPE 網格圖
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mape_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAPE (%)')
    for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
        plt.text(lon, lat, f'{int(round(mape_matrix.flatten()[i]))}', ha='center', va='center', color='black', fontsize=7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAPE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mape.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mape_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAPE (%)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAPE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mape_clean.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 繪製 SMAPE 網格圖（標示整數）
    if smape_matrix is not None:
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(longitudes, latitudes, c=smape_matrix.flatten(), cmap=trunc_cmap, marker='o')
        plt.colorbar(scatter, label='SMAPE (%)')
        for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
            plt.text(lon, lat, f'{int(round(smape_matrix.flatten()[i]))}', ha='center', va='center', color='black', fontsize=7)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Grid with SMAPE")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_smape.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(longitudes, latitudes, c=smape_matrix.flatten(), cmap=trunc_cmap, marker='o')
        plt.colorbar(scatter, label='SMAPE (%)')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Grid with SMAPE")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_smape_clean.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # 更新表格，新增 SMAPE
    table_data = {
        'Grid Index': [f'[{i},{j}]' for i in range(H) for j in range(W)],
        'Longitude': longitudes,
        'Latitude': latitudes,
        'MSE': mse_matrix.flatten(),
        'MAE': mae_matrix.flatten(),
        'MAPE (%)': mape_matrix.flatten()
    }
    if smape_matrix is not None:
        table_data['SMAPE (%)'] = smape_matrix.flatten()
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'mse_mae_mape_smape_per_coordinate.csv'), index=False)
    df.to_excel(os.path.join(save_dir, 'mse_mae_mape_smape_per_coordinate.xlsx'), index=False)

def compute_rgb_mean_std(dataset, sample_count=100):
    """
    遍歷部分資料集（例如目標網格轉換後的 RGB 熱力圖），
    計算所有圖像每個通道的平均值與標準差。
    """
    all_pixels = []
    total_samples = min(len(dataset), sample_count)
    
    for idx in range(total_samples):
        _, target, _ = dataset[idx]  # 解包三個值，僅使用 target
        grid = target.squeeze().cpu().numpy()  # (H, W)
        vmin, vmax = grid.min(), grid.max()
        norm_grid = (grid - vmin) / (vmax - vmin + 1e-8)
        rgb = (plt.cm.viridis(norm_grid)[..., :3] * 255).astype(np.uint8)
        rgb_normalized = rgb.astype(np.float32) / 255.0
        all_pixels.append(rgb_normalized.reshape(-1, 3))
    
    all_pixels = np.concatenate(all_pixels, axis=0)
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)
    return mean.tolist(), std.tolist()

def train_ddpm(diffusion: DDPM3D, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 20, lr: float = 1e-4, device: str = 'cuda', 
               patience: int = 3, weight_decay: float = 1e-6, 
               save_dir: str = r"C:\thesis\code\result_ddpm_hierarchical",
               checkpoint_interval: int = 5) -> DDPM3D:
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    diffusion.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    lr_history = []
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        diffusion.train()
        total_train_loss = 0
        for cond, target, extra_cond in train_loader:
            cond, target, extra_cond = cond.to(device), target.to(device), extra_cond.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
            loss = diffusion.p_losses(cond, target, extra_cond, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for cond, target, extra_cond in val_loader:
                cond, target, extra_cond = cond.to(device), target.to(device), extra_cond.to(device)
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                loss = diffusion.p_losses(cond, target, extra_cond, t)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {current_lr:.8f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': diffusion.state_dict(),
                'learning_rate': current_lr,
            }, os.path.join(save_dir, 'best_model.pth'))
            logging.info(f"保存最佳模型，驗證損失: {best_val_loss:.4f}, 學習率: {current_lr:.8f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rate': current_lr,
            }, checkpoint_path)
            logging.info(f"在 epoch {epoch + 1} 保存檢查點，學習率: {current_lr:.8f}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_history) + 1), lr_history, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'lr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return diffusion
# --------------------------------------
# 複合模型
# --------------------------------------
def generate_rainy_simulations(model, dataset, device, num_samples=100):
    """
    生成雨天模擬數據。
    
    Args:
        model: 已載入的基礎模型 (DDPM3D)。
        dataset: PeopleFlowDatasetCondition 實例。
        device: 計算設備 ('cuda' 或 'cpu')。
        num_samples: 生成的樣本數量。
    
    Returns:
        simulated_flow: 模擬的人流數據 (tensor)。
        actual_flow: 實際的人流數據 (tensor)。
        rainy_indices: 雨天數據的索引。
    """
    model.eval()
    simulated_flow = []
    actual_flow = []
    rainy_indices = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            condition, target, extra = dataset[idx]
            if extra[dataset.extra_columns.index('rain')] > 0:  # 假設 'rain' 是雨量欄位
                rainy_indices.append(idx)
                condition = condition.unsqueeze(0).to(device)  # [1, C, T, H, W]
                extra = torch.tensor(extra, dtype=torch.float32).unsqueeze(0).to(device)
                pred = model.sample(condition, extra)  # 假設 DDPM3D 有 sample 方法
                simulated_flow.append(pred.cpu())
                actual_flow.append(target.unsqueeze(0))

            if len(rainy_indices) >= num_samples:
                break

    simulated_flow = torch.cat(simulated_flow, dim=0)
    actual_flow = torch.cat(actual_flow, dim=0)
    return simulated_flow, actual_flow, rainy_indices

def compute_and_save_offset(simulated_flow, actual_flow, save_path="C:\thesis\code\result_ddpm_hierarchical\rainy_offsets.npy"):
    """
    計算並儲存雨天模擬數據與實際數據的偏移量。
    
    Args:
        simulated_flow: 模擬的人流數據 (tensor)。
        actual_flow: 實際的人流數據 (tensor)。
        save_path: 偏移量儲存路徑。
    
    Returns:
        offsets: 計算出的偏移量 (numpy array)。
    """
    offsets = (actual_flow - simulated_flow).numpy()  # 轉換為 numpy
    np.save(save_path, offsets)
    return offsets

class RainyOffsetDataset(PeopleFlowDatasetCondition):
    """
    雨天偏移數據集，基於偏移量數據進行訓練。
    """
    def __init__(self, csv_path, offset_path, H, W, condition_length, prediction_length, normalize=True):
        super().__init__(csv_path, H, W, condition_length, prediction_length, normalize)
        self.offsets = np.load(offset_path)
        
    def __getitem__(self, idx):
        condition, _, extra = super().__getitem__(idx)
        offset = torch.tensor(self.offsets[idx], dtype=torch.float32)
        return condition, offset, extra

def evaluate_two_stage(base_model, offset_model, dataset, device, save_dir="C:\thesis\code\result_ddpm_hierarchical\two_stage_results"):
    """
    執行兩階段預測並評估。
    
    Args:
        base_model: 基礎模型 (DDPM3D)。
        offset_model: 偏移模型 (DDPM3D)。
        dataset: PeopleFlowDatasetCondition 實例。
        device: 計算設備。
        save_dir: 結果儲存目錄。
    
    Returns:
        one_stage_metrics: 一段式評估指標。
        two_stage_metrics: 兩段式評估指標。
    """
    base_model.eval()
    offset_model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    one_stage_preds = []
    two_stage_preds = []
    targets = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            condition, target, extra = dataset[idx]
            condition = condition.unsqueeze(0).to(device)
            extra = torch.tensor(extra, dtype=torch.float32).unsqueeze(0).to(device)
            target = target.unsqueeze(0)
            
            # 一段式預測
            base_pred = base_model.sample(condition, extra)
            one_stage_preds.append(base_pred.cpu())
            
            # 兩段式預測：基礎預測 + 偏移量
            offset_pred = offset_model.sample(condition, extra)
            two_stage_pred = base_pred + offset_pred
            two_stage_preds.append(two_stage_pred.cpu())
            
            targets.append(target)
    
    one_stage_preds = torch.cat(one_stage_preds, dim=0)
    two_stage_preds = torch.cat(two_stage_preds, dim=0)
    targets = torch.cat(targets, dim=0)
    
    # 計算指標（這裡簡單實現 MSE 和 MAE，您可根據需要擴展）
    one_stage_mse = torch.mean((one_stage_preds - targets) ** 2).item()
    two_stage_mse = torch.mean((two_stage_preds - targets) ** 2).item()
    one_stage_mae = torch.mean(torch.abs(one_stage_preds - targets)).item()
    two_stage_mae = torch.mean(torch.abs(two_stage_preds - targets)).item()
    
    one_stage_metrics = {'mse': one_stage_mse, 'mae': one_stage_mae}
    two_stage_metrics = {'mse': two_stage_mse, 'mae': two_stage_mae}
    
    return one_stage_metrics, two_stage_metrics

@torch.no_grad()
def evaluate_model(diffusion: DDPM3D, dataset: Dataset, device: str = 'cuda', 
                   max_samples: int = 100, save_dir: str = r"C:\\thesis\\code\\result_ddpm_hierarchical",
                   sample_idx: int = 0) -> dict:
    """
    評估模型，並生成視覺化圖表，新增 FID 計算邏輯。
    
    Args:
        diffusion: 訓練好的 DDPM 模型
        dataset: 驗證或測試數據集
        device: 設備
        max_samples: 最大評估樣本數
        save_dir: 結果存檔目錄
        sample_idx: 指定視覺化的樣本索引
    Returns:
        包含 MSE、MAE、MAPE、SMAPE 和 FID 的評估指標字典
    """
    diffusion.eval()
    metrics = {'mse': 0.0, 'mae': 0.0, 'mape': 0.0, 'smape': 0.0, 'fid': 0.0}
    N = min(len(dataset), max_samples)
    sample_indices = random.sample(range(len(dataset)), N)
    
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    H, W = base_dataset.H, base_dataset.W
    pred_length = base_dataset.prediction_length
    
    mean_val = base_dataset.mean_val.to(device)
    std_val = base_dataset.std_val.to(device)
    
    generated_batch = torch.zeros(N, 1, pred_length, H, W, device=device)
    target_batch = torch.zeros(N, 1, pred_length, H, W, device=device)
    
    for i, idx in enumerate(sample_indices):
        cond, target, extra_cond = dataset[idx]  # 解包三個值
        cond = cond.to(device)
        target = target.to(device)
        extra_cond = extra_cond.to(device)
        
        # 準備條件和目標的形狀
        target = target.unsqueeze(2)  # (1, 1, 1, H, W)
        
        # 生成重建結果
        x_recon = diffusion.p_sample_loop(target.shape, cond, extra_cond)
        x_recon_original = x_recon * std_val + mean_val
        target_original = target * std_val + mean_val
        
        generated_batch[i] = x_recon_original
        target_batch[i] = target_original
        
        mse = F.mse_loss(x_recon_original, target_original).item()
        mae = F.l1_loss(x_recon_original, target_original).item()
        mape = torch.mean(torch.abs((target_original - x_recon_original) / (target_original + 1e-10))) * 100
        smape = torch.mean(torch.abs(x_recon_original - target_original) / 
                          (torch.abs(target_original) + torch.abs(x_recon_original) + 1e-10)) * 100
        
        metrics['mse'] += mse
        metrics['mae'] += mae
        metrics['mape'] += mape.item()
        metrics['smape'] += smape.item()
    
    metrics['mse'] /= N
    metrics['mae'] /= N
    metrics['mape'] /= N
    metrics['smape'] /= N

    # 轉換為 RGB 熱力圖，並儲存圖檔
    os.makedirs(save_dir, exist_ok=True)
    pred_images = []
    real_images = []

    # 計算全局範圍以統一色階
    all_pred = generated_batch[:, 0].cpu().numpy().flatten()
    all_real = target_batch[:, 0].cpu().numpy().flatten()
    global_min = min(all_pred.min(), all_real.min())
    global_max = max(all_pred.max(), all_real.max())

    for i in range(N):
        for t in range(pred_length):
            pred_arr = generated_batch[i, 0, t].cpu().numpy()  # (H, W)
            real_arr = target_batch[i, 0, t].cpu().numpy()     # (H, W)
            
            # 使用全局範圍正規化到 [0, 1]
            pred_norm = (pred_arr - global_min) / (global_max - global_min + 1e-8)
            real_norm = (real_arr - global_min) / (global_max - global_min + 1e-8)
            
            # 創建並排的熱力圖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
            
            # 預測熱力圖
            im1 = ax1.imshow(pred_norm, cmap='viridis', interpolation='bilinear')
            ax1.set_title(f"Predicted (Sample {i}, t={t})")
            ax1.set_xlabel("W")
            ax1.set_ylabel("H")
            
            # 真實熱力圖
            im2 = ax2.imshow(real_norm, cmap='viridis', interpolation='bilinear')
            ax2.set_title(f"Real (Sample {i}, t={t})")
            ax2.set_xlabel("W")
            
            # 添加共享色條
            fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', label='Normalized Value')
            
            # 保存合併的熱力圖
            if pred_length == 1:
                filename_combined = f"sample{i}_heatmap_combined.png"
            else:
                filename_combined = f"sample{i}_t{t}_heatmap_combined.png"
            plt.savefig(os.path.join(save_dir, filename_combined), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成獨立的 RGB 圖像以供 FID 計算
            pred_rgb = (plt.cm.viridis(pred_norm)[..., :3] * 255).astype(np.uint8)
            real_rgb = (plt.cm.viridis(real_norm)[..., :3] * 255).astype(np.uint8)
            
            # 保存獨立的熱力圖
            if pred_length == 1:
                filename_pred = f"sample{i}_pred_heatmap.png"
                filename_real = f"sample{i}_real_heatmap.png"
            else:
                filename_pred = f"sample{i}_t{t}_pred_heatmap.png"
                filename_real = f"sample{i}_t{t}_real_heatmap.png"
            Image.fromarray(pred_rgb).save(os.path.join(save_dir, filename_pred))
            Image.fromarray(real_rgb).save(os.path.join(save_dir, filename_real))
            
            pred_images.append(Image.fromarray(pred_rgb))
            real_images.append(Image.fromarray(real_rgb))
    
    # 在特徵提取前檢查樣本數
    print(f"Number of pred_images: {len(pred_images)}")
    print(f"Number of real_images: {len(real_images)}")
    if len(pred_images) < 2 or len(real_images) < 2:
        raise ValueError("樣本數不足以計算 FID，至少需要 2 個樣本")
    
    new_mean, new_std = compute_rgb_mean_std(dataset, sample_count=100)
    
    # 使用 InceptionV3 提取特徵
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    inception_model.fc = torch.nn.Identity()  # 移除分類層
    inception_model.AuxLogits = None  # 手動移除輔助分支
    inception_model.to(device)
    inception_model.eval()
    
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(new_mean, new_std)
    ])
    
    pred_tensors = [inception_transform(img).to(device) for img in pred_images]
    real_tensors = [inception_transform(img).to(device) for img in real_images]
    pred_tensor_batch = torch.stack(pred_tensors)
    real_tensor_batch = torch.stack(real_tensors)
    
    with torch.no_grad():
        pred_features = inception_model(pred_tensor_batch)
        real_features = inception_model(real_tensor_batch)
    
    pred_features_np = pred_features.cpu().numpy()
    real_features_np = real_features.cpu().numpy()

    mu_pred = np.mean(pred_features_np, axis=0)
    mu_real = np.mean(real_features_np, axis=0)
    sigma_pred = np.cov(pred_features_np, rowvar=False)
    sigma_real = np.cov(real_features_np, rowvar=False)
    
    covmean, _ = scipy.linalg.sqrtm(sigma_pred.dot(sigma_real), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu_pred - mu_real)**2) + np.trace(sigma_pred + sigma_real - 2 * covmean)
    metrics['fid'] = fid
    
    # 計算每個網格點的誤差矩陣
    error_matrix_mse = (generated_batch - target_batch) ** 2
    mse_matrix = torch.mean(error_matrix_mse, dim=(0, 2)).cpu().numpy()[0]  # (H, W)
    
    error_matrix_mae = torch.abs(generated_batch - target_batch)
    mae_matrix = torch.mean(error_matrix_mae, dim=(0, 2)).cpu().numpy()[0]  # (H, W)
    
    mape_matrix = torch.mean(torch.abs((target_batch - generated_batch) / (target_batch + 1e-10)), 
                            dim=(0, 2)).cpu().numpy()[0] * 100  # (H, W)
    
    smape_matrix = torch.mean(torch.abs(generated_batch - target_batch) / 
                             (torch.abs(target_batch) + torch.abs(generated_batch) + 1e-10), 
                             dim=(0, 2)).cpu().numpy()[0] * 100  # (H, W)
    
    # 儲存誤差數據到表格，包含 FID
    table_data = {
        'Grid Index': [f'[{i},{j}]' for i in range(H) for j in range(W)],
        'Longitude': [parse_lat_lon(col)[0] for col in base_dataset.sorted_flow_columns],
        'Latitude': [parse_lat_lon(col)[1] for col in base_dataset.sorted_flow_columns],
        'MSE': mse_matrix.flatten(),
        'MAE': mae_matrix.flatten(),
        'MAPE (%)': mape_matrix.flatten(),
        'SMAPE (%)': smape_matrix.flatten(),
        'FID': [metrics['fid']] * (H * W)  # FID 作為全域指標，應用於所有網格
    }
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'mse_mae_mape_smape_fid_per_coordinate.csv'), index=False)
    df.to_excel(os.path.join(save_dir, 'mse_mae_mape_smape_fid_per_coordinate.xlsx'), index=False)
    
    # 視覺化
    plot_grid_with_error(base_dataset.sorted_flow_columns, H, W, mse_matrix, mae_matrix, mape_matrix, save_dir, smape_matrix)
    visualize_predictions(None, generated_batch, target_batch, sample_idx, save_dir)
    
    # 儲存評估指標
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Evaluation Metrics (computed on {N} samples):\n")
        f.write(f"Reconstruction MSE: {metrics['mse']:.6f}\n")
        f.write(f"Reconstruction MAE: {metrics['mae']:.6f}\n")
        f.write(f"Reconstruction MAPE: {metrics['mape']:.6f}%\n")
        f.write(f"Reconstruction SMAPE: {metrics['smape']:.6f}%\n")
        f.write(f"Reconstruction FID: {metrics['fid']:.6f}\n")
    
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump({
            "mse": metrics['mse'], 
            "mae": metrics['mae'], 
            "mape": metrics['mape'], 
            "smape": metrics['smape'],
            "fid": metrics['fid'],
            "sample_size": N, 
            "timestamp": pd.Timestamp.now().isoformat()
        }, f, indent=4)
    
    return metrics

if __name__ == "__main__":
    # 設定參數
    H, W = 21, 21
    condition_length, prediction_length = 8, 1
    batch_size, epochs, lr, timesteps = 32, 20, 1e-4, 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 載入數據集
    dataset = PeopleFlowDatasetCondition(
        csv_path=r"C:\thesis\code\Taipei_CF\all_merged.csv",
        H=H, W=W, condition_length=condition_length, prediction_length=prediction_length,
        normalize=True
    )
    
    # 載入基礎模型
    base_model_path = r"C:\thesis\code\result_ddpm_condition\best_model.pth"
    unet_base = UNet3D(in_channels=1, base_channels=64, time_emb_dim=128)
    base_model = DDPM3D(model=unet_base, timesteps=timesteps, device=device, condition_dim=len(dataset.extra_columns))
    base_model.load_state_dict(torch.load(base_model_path)['model_state_dict'])
    base_model.to(device)
    
    # 生成雨天模擬數據
    simulated_flow, actual_flow, rainy_indices = generate_rainy_simulations(base_model, dataset, device)
    
    # 計算並儲存偏移量
    offsets = compute_and_save_offset(simulated_flow, actual_flow)
    
    # 訓練雨天偏移模型
    rainy_dataset = RainyOffsetDataset(
        csv_path=r"C:\thesis\code\Taipei_CF\all_merged.csv",
        offset_path="rainy_offsets.npy",
        H=H, W=W, condition_length=condition_length, prediction_length=prediction_length
    )
    train_end = int(0.7 * len(rainy_dataset))
    train_loader = DataLoader(Subset(rainy_dataset, range(train_end)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(rainy_dataset, range(train_end, len(rainy_dataset))), batch_size=batch_size, shuffle=False)
    unet_offset = UNet3D(in_channels=1, base_channels=64, time_emb_dim=128)
    offset_model = DDPM3D(model=unet_offset, timesteps=timesteps, device=device, condition_dim=len(dataset.extra_columns))
    trained_offset_model = train_ddpm(offset_model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
    
    # 評估兩階段方法
    one_stage_metrics, two_stage_metrics = evaluate_two_stage(base_model, trained_offset_model, dataset, device)
    print(f"一段式 MSE: {one_stage_metrics['mse']:.6f}, 兩段式 MSE: {two_stage_metrics['mse']:.6f}")
    print(f"一段式 MAE: {one_stage_metrics['mae']:.6f}, 兩段式 MAE: {two_stage_metrics['mae']:.6f}")

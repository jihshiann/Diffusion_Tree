import os
import re
import math
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

# 設定 logging 等級為 INFO，方便查看訓練過程
logging.basicConfig(level=logging.INFO)

# 定義常數：時間嵌入的維度
TIME_EMB_DIM = 128

# ### 數據處理相關函數
def parse_lat_lon(column_name: str) -> tuple[float, float]:
    """從欄位名稱中解析經緯度座標。
    
    Args:
        column_name (str): 包含經緯度的欄位名稱，例如 '(121.5, 25.0)'。
    
    Returns:
        tuple[float, float]: 經度 (lon) 和緯度 (lat)。

    Raises:
        ValueError: 若欄位名稱格式無效。
    """
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class GridData:
    """從 CSV 檔案中讀取數據並映射到指定大小的網格。
    
    Attributes:
        df (pd.DataFrame): 原始數據表格。
        data (torch.Tensor): 重塑後的網格數據，形狀為 (n_samples, H, W)。
        mean_val (float): 數據均值，用於正規化。
        std_val (float): 數據標準差，用於正規化。
        H (int): 網格高度。
        W (int): 網格寬度。
    """
    def __init__(self, csv_path: str, H: int, W: int, normalize: bool = True, debug: bool = False):
        # 讀取 CSV 文件
        self.df = pd.read_csv(csv_path)
        
        # 找出包含經緯度的欄位名稱（名稱中包含括號）
        flow_columns = [c for c in self.df.columns if '(' in c and ')' in c]
        
        # 解析欄位名稱，獲取 (名稱, 經度, 緯度) 三元組
        column_info = [(col, *parse_lat_lon(col)) for col in flow_columns]
        
        # 提取經緯度座標，組成 numpy 陣列，順序為 (lon, lat)
        coords = np.array([(lon, lat) for _, lon, lat in column_info])
        
        # 計算中心點：使用所有座標的平均值作為中心
        mean_lon, mean_lat = np.mean(coords, axis=0)
        distances_to_center = np.sqrt((coords[:, 0] - mean_lon)**2 + (coords[:, 1] - mean_lat)**2)
        central_idx = np.argmin(distances_to_center)  # 找到距離中心最近的座標索引
        central_coord = coords[central_idx]
        
        # 初始化 21x21 網格，中心點位於 (10, 10)，未分配位置填充 -1
        grid_size = 21
        grid = np.full((grid_size, grid_size), -1, dtype=int)
        central_row, central_col = 10, 10
        grid[central_row, central_col] = central_idx
        
        # 計算網格步長：根據經緯度差異的中位數確定
        lon_diffs = np.diff(np.sort(coords[:, 0]))
        lat_diffs = np.diff(np.sort(coords[:, 1]))
        lon_step = np.median(lon_diffs[lon_diffs > 0]) if len(lon_diffs) > 0 else 0.005
        lat_step = np.median(lat_diffs[lat_diffs > 0]) if len(lat_diffs) > 0 else 0.005
        
        # 分配剩餘座標到網格
        available_indices = list(range(len(coords)))
        available_indices.remove(central_idx)
        grid_positions = []
        
        # 以曼哈頓距離層級 (k) 從中心向外分配位置
        for k in range(11):
            for r in range(max(0, 10 - k), min(21, 10 + k + 1)):
                for c in range(max(0, 10 - k), min(21, 10 + k + 1)):
                    if max(abs(r - 10), abs(c - 10)) == k:
                        grid_positions.append((r, c))
        
        # 為每個網格位置選擇最接近的座標
        for r, c in grid_positions:
            if grid[r, c] != -1:
                continue  # 跳過已分配的中心點
            # 根據網格位置計算目標經緯度
            target_lon = central_coord[0] + (c - central_col) * lon_step
            target_lat = central_coord[1] - (r - central_row) * lat_step
            
            # 設置經緯度約束條件
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
            
            # 篩選符合約束條件的座標
            filtered_indices = [idx for idx in available_indices if
                                (lon_constraint is None or lon_constraint(coords[idx][0])) and
                                (lat_constraint is None or lat_constraint(coords[idx][1]))]
            if filtered_indices:
                # 若有符合條件的座標，選擇距離目標最近的
                distances = np.sqrt((coords[filtered_indices, 0] - target_lon)**2 +
                                    (coords[filtered_indices, 1] - target_lat)**2)
                closest_idx = filtered_indices[np.argmin(distances)]
            else:
                # 若無符合條件的，從剩餘座標中選擇最近的
                distances = np.sqrt((coords[available_indices, 0] - target_lon)**2 +
                                    (coords[available_indices, 1] - target_lat)**2)
                closest_idx = available_indices[np.argmin(distances)]
            
            grid[r, c] = closest_idx
            available_indices.remove(closest_idx)
        
        # 根據網格順序重新排列欄位並提取數據
        sorted_indices = grid.flatten()
        self.sorted_flow_columns = [column_info[idx][0] for idx in sorted_indices]
        flow_values = self.df[self.sorted_flow_columns].values.reshape(-1, H, W).astype(np.float32)
        self.data = torch.from_numpy(flow_values)
        
        # 若需要正規化，計算均值和標準差並應用
        if normalize:
            self.mean_val = self.data.mean()
            self.std_val = self.data.std() + 1e-5  # 避免除以零
            self.data = (self.data - self.mean_val) / self.std_val
        else:
            self.mean_val = 0
            self.std_val = 1
        self.H = H
        self.W = W

class PeopleFlowDatasetPerCell(Dataset):
    """為每個網格單元生成數據集，輸入為整個 21x21 網格的條件序列。
    
    Attributes:
        grid_data (GridData): 網格數據對象。
        condition_length (int): 條件序列長度。
        prediction_length (int): 預測長度。
        i (int): 目標單元的行索引。
        j (int): 目標單元的列索引。
        max_index (int): 可用樣本數。
    """
    def __init__(self, grid_data: GridData, condition_length: int, prediction_length: int, i: int, j: int):
        self.grid_data = grid_data
        self.condition_length = condition_length
        self.prediction_length = prediction_length
        self.i = i
        self.j = j
        # 計算最大索引，確保不越界
        self.max_index = grid_data.data.shape[0] - condition_length - prediction_length + 1

    def __len__(self):
        return self.max_index

    def __getitem__(self, idx: int):
        # 條件序列：(1, condition_length, H, W)
        condition = self.grid_data.data[idx:idx + self.condition_length].unsqueeze(0)
        # 目標序列：(prediction_length,)，指定單元的連續時間步
        target = self.grid_data.data[idx + self.condition_length:idx + self.condition_length + self.prediction_length, self.i, self.j]
        return condition, target  # target 形狀為 (prediction_length,)

class PeopleFlowDatasetForEval(Dataset):
    """為評估生成數據集，返回整個 21x21 網格的目標值。
    
    Attributes:
        grid_data (GridData): 網格數據對象。
        condition_length (int): 條件序列長度。
        prediction_length (int): 預測長度。
        max_index (int): 可用樣本數。
    """
    def __init__(self, grid_data: GridData, condition_length: int, prediction_length: int):
        self.grid_data = grid_data
        self.condition_length = condition_length
        self.prediction_length = prediction_length
        # 計算最大索引，確保不越界
        self.max_index = grid_data.data.shape[0] - condition_length - prediction_length + 1

    def __len__(self):
        return self.max_index

    def __getitem__(self, idx: int):
        # 條件序列：(1, condition_length, H, W)
        condition = self.grid_data.data[idx:idx + self.condition_length].unsqueeze(0)
        # 目標序列：(prediction_length, H, W)
        target = self.grid_data.data[idx + self.condition_length:idx + self.condition_length + self.prediction_length]
        return condition, target  # target 形狀為 (prediction_length, H, W)

# ### 模型定義
class DoubleConv3D(nn.Module):
    """3D雙卷積模塊"""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ScalarPredictor(nn.Module):
    """輕量級 CNN 模型，預測單一標量，適應 21×21 輸入
    
    Args:
        time_emb_dim (int): 時間嵌入維度，默認為 128。
        base_channels (int): 基礎通道數，設為 64。
    """
    def __init__(self, time_emb_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.enc1 = DoubleConv3D(1, base_channels)  # (batch, 1, 8, 21, 21) -> (batch, 64, 8, 21, 21)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # -> (batch, 64, 4, 10, 10)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)  # -> (batch, 128, 4, 10, 10)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # -> (batch, 128, 2, 5, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # -> (batch, 128, 1, 1, 1)
        feature_size = base_channels * 2  # 128
        self.fc = nn.Linear(feature_size + 1 + time_emb_dim, 1)  # 直接映射到單一標量

    def forward(self, condition: torch.Tensor, x_t: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(condition)
        e2 = self.enc2(self.pool1(e1))
        pooled = self.adaptive_pool(self.pool2(e2)).view(condition.size(0), -1)
        # 確保 x_t 是二維張量 (batch_size, 1)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        combined = torch.cat([pooled, x_t, t_emb], dim=1)
        out = self.fc(combined)
        return out

class DDPMPerCell(nn.Module):
    """每個網格單元的 DDPM 模型，實現噪聲添加和去噪過程"""
    """
    初始化 DDPM (Denoising Diffusion Probabilistic Model) 模型，設置擴散過程的基本參數。

    Args:
        model (nn.Module): 用於去噪的神經網路模型，負責預測每個時間步的噪聲成分。
        timesteps (int, optional): 擴散過程的總步數，預設為 1000。表示前向加噪和反向去噪的總步數。
        beta_start (float, optional): 噪聲強度的起始值，預設為 1e-4。控制第一步添加的噪聲量，通常較小以保留更多原始數據。
        beta_end (float, optional): 噪聲強度的結束值，預設為 0.02。控制最後一步添加的噪聲量，通常較大以接近純噪聲。
        device (str, optional): 計算設備，預設為 'cuda'。指定模型和張量運行的硬體（例如 'cuda' 或 'cpu'）。

    Attributes:
        self.model: 儲存傳入的去噪模型。
        self.timesteps: 儲存擴散步數。
        self.device: 儲存設備類型。
        self.betas: 形狀為 (timesteps,) 的張量，從 beta_start 到 beta_end 線性生成，控制每一步的噪聲強度。
        self.alphas: 形狀為 (timesteps,) 的張量，計算為 1 - betas，表示每一步保留的原始信號比例。
        self.alphas_cumprod: 形狀為 (timesteps,) 的張量，alphas 的累積乘積，表示第 t 步後保留的原始信號總比例。
        self.sqrt_alphas_cumprod: 形狀為 (timesteps,) 的張量，alphas_cumprod 的平方根，用於前向擴散過程的計算。
        self.sqrt_one_minus_alphas_cumprod: 形狀為 (timesteps,) 的張量，1 - alphas_cumprod 的平方根，控制噪聲的比例。
        self.half_dim: 時間嵌入維度的一半，預設為 TIME_EMB_DIM // 2，用於生成頻率因子。
        self.freq_factor: 形狀為 (half_dim,) 的張量，用於時間嵌入的頻率因子，將時間步 t 轉換為正弦和餘弦函數的嵌入向量。
    """
    def __init__(self, model: nn.Module, timesteps: int = 1000, 
                 beta_start: float = 1e-4, beta_end: float = 0.02, device: str = 'cuda'):
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

    # 生成時間嵌入，將時間步 t 轉換為向量，讓模型感知當前的擴散步驟
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        emb = t[:, None] * self.freq_factor.to(t.device)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    # 執行前向擴散過程，將原始數據 x0 在時間步 t 加入噪聲，生成含噪數據 x_t
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, *(1 for _ in x0.shape[1:]))
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *(1 for _ in x0.shape[1:]))
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    # 計算訓練損失，訓練模型預測噪聲
    def p_losses(self, condition: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(target)
        x_noisy = self.q_sample(target, t, noise=noise)
        time_emb = self.get_time_embedding(t).to(self.device)
        pred_noise = self.model(condition, x_noisy, time_emb)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    # 執行反向去噪過程的單步操作，從含噪數據 x_t 生成前一步數據 x_{t-1}
    def p_sample(self, condition: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[t].view(-1, *(1 for _ in x_t.shape[1:]))
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t]).view(-1, *(1 for _ in x_t.shape[1:]))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *(1 for _ in x_t.shape[1:]))
        time_emb = self.get_time_embedding(t).to(self.device)
        eps_theta = self.model(condition, x_t, time_emb)
        x_t_minus_1 = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)
        mask = (t > 0).float().view(-1, *(1 for _ in x_t.shape[1:]))
        sigma_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        return x_t_minus_1 + mask * sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(self, condition: torch.Tensor, shape: tuple, prediction_length: int) -> torch.Tensor:
        batch_size = shape[0]
        predictions = []
        x = torch.randn((batch_size, 1), device=self.device)  # 初始隨機噪聲
        if condition.dim() == 4:
            condition = condition.unsqueeze(1)

        for t_idx in range(prediction_length):
            # 逐步去噪生成當前時間步
            for i in reversed(range(self.timesteps)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                x = self.p_sample(condition, x, t)
            predictions.append(x.clone())  # 保存當前預測

            # 更新條件：將當前預測加入條件序列（模擬自迴歸）
            if t_idx < prediction_length - 1:
                new_condition_slice = x.view(batch_size, 1, 1, 1)  # 假設單一值擴展為條件
                condition = torch.cat([condition[:, :, 1:], new_condition_slice], dim=2)  # 移除最早時間步，加入新預測
                x = torch.randn((batch_size, 1), device=self.device)  # 重置噪聲以預測下一個時間步

        return torch.stack(predictions, dim=1)  # 形狀：(batch_size, prediction_length)

# ### 繪製網格誤差圖的函數
def plot_grid_with_error(sorted_flow_columns: list, H: int, W: int, mse_matrix: np.ndarray, mae_matrix: np.ndarray, mape_matrix: np.ndarray, 
                         save_dir: str = r"C:\thesis\code\result_ddpm_perCell\evaluate", smape_matrix: np.ndarray = None):
    """
    繪製網格圖，顯示每個網格點的誤差（MSE、MAE、MAPE 和 SMAPE），並將結果存成圖與表格。
    
    Args:
        sorted_flow_columns (list): 經緯度欄位的排序列表。
        H (int): 網格高度。
        W (int): 網格寬度。
        mse_matrix (np.ndarray): 每個網格點的 MSE 矩陣，形狀為 (H, W)。
        mae_matrix (np.ndarray): 每個網格點的 MAE 矩陣，形狀為 (H, W)。
        mape_matrix (np.ndarray): 每個網格點的 MAPE 矩陣，形狀為 (H, W)。
        save_dir (str): 存檔路徑。
        smape_matrix (np.ndarray, optional): 每個網格點的 SMAPE 矩陣，形狀為 (H, W)。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 解析經緯度
    locations = [parse_lat_lon(col) for col in sorted_flow_columns]
    longitudes, latitudes = zip(*locations)
    
    # 定義顏色映射
    orig_cmap = plt.get_cmap('OrRd')
    trunc_cmap = truncate_colormap(orig_cmap, 0.3, 1.0)
    
    # 繪製 MSE 網格圖（保留 colorbar）
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mse_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MSE')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MSE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mse.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 繪製 MAE 網格
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
    plt.colorbar(scatter, label='MAPE')
    for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
        plt.text(lon, lat, f'{int(round(mape_matrix.flatten()[i]))}', ha='center', va='center', color='black', fontsize=7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAPE (%)")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mape.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mape_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAPE')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAPE (%)")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mape_clean.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    

    # 繪製 SMAPE 網格圖
    if smape_matrix is not None:
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(longitudes, latitudes, c=smape_matrix.flatten(), cmap=trunc_cmap, marker='o')
        plt.colorbar(scatter, label='SMAPE')
        for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
            plt.text(lon, lat, f'{int(round(smape_matrix.flatten()[i]))}', ha='center', va='center', color='black', fontsize=7)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Grid with SMAPE (%)")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_smape.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(longitudes, latitudes, c=smape_matrix.flatten(), cmap=trunc_cmap, marker='o')
        plt.colorbar(scatter, label='SMAPE')
        plt.ylabel("Latitude")
        plt.title("Grid with SMAPE (%)")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_smape_simple.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # 保存表格
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
    
def visualize_predictions(cond, generated, target, sample_idx: int = None, 
                         save_dir: str = r"C:\thesis\code\result_ddpm_perCell\evaluate", 
                         mse_matrix: Optional[np.ndarray] = None, 
                         mae_matrix: Optional[np.ndarray] = None, 
                         mape_matrix: Optional[np.ndarray] = None, 
                         smape_matrix: Optional[np.ndarray] = None):
    """
    視覺化所有樣本和時間步平均後的生成預測值與真實值的比較，將每個指標儲存為獨立圖形。
    MAE、MAPE 和 SMAPE 圖在網格上標示整數數值，移除右側尺度，利用已有誤差矩陣。
    
    Args:
        cond: 條件張量，此處未使用。
        generated: 生成的預測值，形狀 (N, 1, prediction_length, H, W)。
        target: 目標值，形狀 (N, 1, prediction_length, H, W)。
        sample_idx: 樣本索引，此處設為 None 表示平均圖。
        save_dir: 保存視覺化結果的目錄。
        mse_matrix: 預計算的 MSE 矩陣，形狀 (H, W)。
        mae_matrix: 預計算的 MAE 矩陣，形狀 (H, W)。
        mape_matrix: 預計算的 MAPE 矩陣，形狀 (H, W)。
        smape_matrix: 預計算的 SMAPE 矩陣，形狀 (H, W)。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 對所有樣本和時間步取平均
    generated_avg = torch.mean(generated, dim=(0, 2)).squeeze(0).cpu().numpy()  # 形狀：(H, W)
    target_avg = torch.mean(target, dim=(0, 2)).squeeze(0).cpu().numpy()        # 形狀：(H, W)
    
    # 使用預計算的誤差矩陣，若未提供則重新計算（保持二維結構）
    if mse_matrix is None:
        mse_matrix = (generated_avg - target_avg) ** 2  # 形狀：(H, W)
    if mae_matrix is None:
        mae_matrix = np.abs(generated_avg - target_avg)  # 形狀：(H, W)
    if mape_matrix is None:
        mape_matrix = np.abs((generated_avg - target_avg) / (target_avg + 1)) * 100  # 形狀：(H, W)
    if smape_matrix is None:
        smape_matrix = np.abs(generated_avg - target_avg) / (np.abs(target_avg) + np.abs(generated_avg) + 1) * 100  # 形狀：(H, W)
    
    mse = np.mean(mse_matrix)  # 用於標題顯示
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
    
    # 子圖 4：MAE 誤差圖
    plt.figure(figsize=(6, 6))
    plt.imshow(mae_matrix, cmap='hot')
    plt.colorbar()
    for i in range(mae_matrix.shape[0]):
        for j in range(mae_matrix.shape[1]):
            plt.text(j, i, f'{int(round(mae_matrix[i, j]))}', ha='center', va='center', color='white', fontsize=4)
    plt.title(f'MAE: {mae:.0f}')
    plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_mae.png'), dpi=300)
    plt.close()
    
    # 子圖 5：MAPE 誤差圖
    plt.figure(figsize=(6, 6))
    plt.imshow(mape_matrix, cmap='hot')
    plt.colorbar()
    for i in range(mape_matrix.shape[0]):
        for j in range(mape_matrix.shape[1]):
            plt.text(j, i, f'{int(round(mape_matrix[i, j]))}', ha='center', va='center', color='white', fontsize=4)
    plt.title(f'MAPE: {mape:.0f}%')
    plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_mape.png'), dpi=300)
    plt.close()
    
    # 子圖 6：SMAPE 誤差圖（標示整數數值，移除 colorbar）
    plt.figure(figsize=(6, 6))
    plt.imshow(smape_matrix, cmap='hot')
    plt.colorbar()
    for i in range(smape_matrix.shape[0]):
        for j in range(smape_matrix.shape[1]):
            plt.text(j, i, f'{int(round(smape_matrix[i, j]))}', ha='center', va='center', color='white', fontsize=4)
    plt.title(f'SMAPE: {smape:.0f}%')
    plt.savefig(os.path.join(save_dir, 'prediction_all_samples_avg_smape.png'), dpi=300)
    plt.close()

def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """
    截斷色彩圖，僅使用指定範圍的色階。
    """
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# ### 訓練與評估函數
def train_ddpm(diffusion: DDPMPerCell, train_loader: DataLoader, val_loader: DataLoader, i: int, j: int,
               start_epoch: int = 0, epochs: int = 20, lr: float = 1e-4, device: str = 'cuda', 
               patience: int = 3, weight_decay: float = 1e-6, 
               save_dir: str = r"C:\thesis\code\result_ddpm_perCell\model", checkpoint_interval: int = 10) -> DDPMPerCell:
    """訓練 DDPM 模型，帶有檢查點保存和恢復功能。
    
    Args:
        diffusion (DDPMPerCell): DDPM 模型。
        train_loader (DataLoader): 訓練數據載入器。
        val_loader (DataLoader): 驗證數據載入器。
        i (int): 網格單元的行索引。
        j (int): 網格單元的列索引。
        start_epoch (int): 起始輪數。
        epochs (int): 總訓練輪數。
        lr (float): 學習率。
        device (str): 運行設備。
        patience (int): 早停耐心值。
        weight_decay (float): 權重衰減。
        save_dir (str): 模型保存目錄。
        checkpoint_interval (int): 檢查點保存間隔。
    
    Returns:
        DDPMPerCell: 訓練完成的模型。
    """
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    diffusion.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{i}_{j}.pth')
    
    # 檢查並恢復檢查點
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_loss = min(val_losses) if val_losses else float('inf')
        logging.info(f"恢復單元 ({i}, {j}) 的訓練，從 epoch {start_epoch} 開始")
    
    for epoch in range(start_epoch, epochs):
        diffusion.train()
        total_train_loss = 0
        for condition, target in train_loader:
            condition, target = condition.to(device), target.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
            loss = diffusion.p_losses(condition, target, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for condition, target in val_loader:
                condition, target = condition.to(device), target.to(device)
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                loss = diffusion.p_losses(condition, target, t)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logging.info(f"單元 ({i}, {j}) Epoch [{epoch+1}/{epochs}] - 訓練損失: {avg_train_loss:.4f}, 驗證損失: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(diffusion.state_dict(), os.path.join(save_dir, f'best_model_{i}_{j}.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"單元 ({i}, {j}) 觸發早停")
                break
        
        # 定期保存檢查點
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'i': i,
                'j': j
            }, checkpoint_path)
            logging.info(f"單元 ({i}, {j}) 在 epoch {epoch + 1} 保存檢查點")
    
    
    # 繪製損失曲線
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f' ({i}, {j}) loss curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_curve_{i}_{j}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return diffusion

def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """
    截斷色彩圖，僅使用指定範圍的色階。
    """
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

@torch.no_grad()
def evaluate_model_per_cell(models: dict, grid_data: GridData, test_dataset: Dataset, device: str = 'cuda', 
                            max_samples: int = 100, save_dir: str = r"C:\thesis\code\result_ddpm_perCell\evaluate") -> dict:
    """
    評估模型在每個網格點的性能，計算 MSE、MAE、MAPE 和 SMAPE。
    
    Args:
        models (dict): 每個網格點的模型字典，鍵為 (i, j)，值為 DDPMPerCell 實例。
        grid_data (GridData): 網格數據，包含均值和標準差。
        test_dataset (Dataset): 測試數據集。
        device (str): 設備（預設為 'cuda'）。
        max_samples (int): 最大樣本數（預設為 100）。
        save_dir (str): 結果儲存路徑。
    
    Returns:
        dict: 包含 MSE、MAE、MAPE 和 SMAPE 的評估指標。
    """
    # 設定日誌基本配置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    metrics = {'mse': 0.0, 'mae': 0.0, 'mape': 0.0, 'smape': 0.0}
    N = min(len(test_dataset), max_samples)
    print(len(test_dataset))
    print(N)
    sample_indices = random.sample(range(len(test_dataset)), N)
    H, W = grid_data.H, grid_data.W
    prediction_length = test_dataset.dataset.prediction_length
    mean_val = grid_data.mean_val.to(device)
    std_val = grid_data.std_val.to(device)
    
    # 初始化累加器
    generated_batch = torch.zeros(N, 1, prediction_length, H, W, device=device)
    target_batch = torch.zeros(N, 1, prediction_length, H, W, device=device)
    mse_sum = torch.zeros(H, W, device=device)
    mae_sum = torch.zeros(H, W, device=device)
    mape_sum = torch.zeros(H, W, device=device)
    smape_sum = torch.zeros(H, W, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for (i, j), model in models.items():
        model.to(device)
        model.eval()

    for k, idx in enumerate(sample_indices):
        logging.info(f"評估進度: {k+1}/{N} 樣本")
        cond, target = test_dataset[idx]
        cond = cond.to(device)
        target = target.to(device)
        predictions = torch.zeros(1, prediction_length, H, W, device=device)
        
        for i in range(H):
            for j in range(W):
                logging.info(f"正在處理單元 ({i}, {j})")
                model = models[(i, j)]
                x_pred = model.p_sample_loop(condition=cond, shape=(1, 1), prediction_length=prediction_length)
                predictions[0, :, i, j] = x_pred[0, :]
        
        x_recon_original = predictions * std_val + mean_val
        target_original = target * std_val + mean_val
        target_original = target_original.view(1, prediction_length, H, W)
        
        # 計算誤差並累加
        error = x_recon_original - target_original
        mse_grid = torch.mean(error ** 2, dim=(0, 1))  # 每個網格的 MSE
        mae_grid = torch.mean(torch.abs(error), dim=(0, 1))  # 每個網格的 MAE
        mape_grid = torch.mean(torch.abs(error / (target_original + 1)), dim=(0, 1)) * 100  # MAPE
        smape_grid = torch.mean(torch.abs(error) / (torch.abs(target_original) + torch.abs(x_recon_original) + 1), dim=(0, 1)) * 100  # SMAPE
        
        mse_sum += mse_grid
        mae_sum += mae_grid
        mape_sum += mape_grid
        smape_sum += smape_grid
        
        generated_batch[k] = x_recon_original
        target_batch[k] = target_original
    
    # 計算整體平均指標
    metrics['mse'] = torch.mean(mse_sum / N).item()
    metrics['mae'] = torch.mean(mae_sum / N).item()
    metrics['mape'] = torch.mean(mape_sum / N).item()
    metrics['smape'] = torch.mean(smape_sum / N).item()
    
    # 計算網格級誤差矩陣
    mse_matrix = (mse_sum / N).cpu().numpy()
    mae_matrix = (mae_sum / N).cpu().numpy()
    mape_matrix = (mape_sum / N).cpu().numpy()
    smape_matrix = (smape_sum / N).cpu().numpy()
    
    # 匯出表格
    table_data = {
        'Grid Index': [f'[{i},{j}]' for i in range(H) for j in range(W)],
        'Longitude': [parse_lat_lon(col)[0] for col in grid_data.sorted_flow_columns],
        'Latitude': [parse_lat_lon(col)[1] for col in grid_data.sorted_flow_columns],
        'MSE': mse_matrix.flatten(),
        'MAE': mae_matrix.flatten(),
        'MAPE (%)': mape_matrix.flatten(),
        'SMAPE (%)': smape_matrix.flatten()
    }
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'mse_mae_mape_smape_per_coordinate.csv'), index=False)
    df.to_excel(os.path.join(save_dir, 'mse_mae_mape_smape_per_coordinate.xlsx'), index=False)
    
    # 繪製網格誤差圖（散點圖）
    plot_grid_with_error(grid_data.sorted_flow_columns, H, W, mse_matrix, mae_matrix, mape_matrix, save_dir, smape_matrix=smape_matrix)
    
    # 繪製所有樣本的平均圖（熱力圖）
    visualize_predictions(None, generated_batch, target_batch, sample_idx=None, save_dir=save_dir,
                          mse_matrix=mse_matrix, mae_matrix=mae_matrix, mape_matrix=mape_matrix, smape_matrix=smape_matrix)
    
    # 保存整體指標
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"重建 MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, MAPE: {metrics['mape']:.6f}, SMAPE: {metrics['smape']:.6f}")
    return metrics

### 主程式
if __name__ == "__main__":
    # 定義超參數
    H, W = 21, 21
    condition_length, prediction_length = 8, 1
    batch_size, epochs, lr, timesteps, patience = 150, 150, 0.0005, 1000, 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = r"C:\thesis\code\result_ddpm_perCell\model"
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 載入數據
    grid_data = GridData(csv_path=r"C:\thesis\code\Taipei_CF\all_merged.csv", H=H, W=W, normalize=True, debug=True)
    
    # 創建用於評估的全局測試數據集
    test_dataset = PeopleFlowDatasetForEval(grid_data, condition_length, prediction_length)
    test_end = int(0.85 * len(test_dataset))
    test_dataset = Subset(test_dataset, range(test_end, len(test_dataset)))
    print(device)

    # 訓練 441 個模型（對應 21x21 網格的每個單元）
    for i in range(H):
        for j in range(W):
            dataset = PeopleFlowDatasetPerCell(grid_data, condition_length, prediction_length, i, j)
            train_end = int(0.7 * len(dataset))
            val_end = int(0.85 * len(dataset))
            train_dataset = Subset(dataset, range(0, train_end))
            val_dataset = Subset(dataset, range(train_end, val_end))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
            # 初始化並訓練模型
            model = ScalarPredictor(time_emb_dim=TIME_EMB_DIM, base_channels=64)
            diffusion = DDPMPerCell(model=model, timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)
            trained_diffusion = train_ddpm(diffusion, train_loader, val_loader, i, j, epochs=epochs, 
                                          lr=lr, device=device, patience=patience, save_dir=save_dir, checkpoint_interval=5)

    # 評估模型
    models = {}
    for i in range(H):
        for j in range(W):
            model = ScalarPredictor(time_emb_dim=TIME_EMB_DIM, base_channels=64)
            diffusion = DDPMPerCell(model=model, timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)
            best_model_path = os.path.join(save_dir, f'best_model_{i}_{j}.pth')
            checkpoint_model_path = os.path.join(save_dir, f'checkpoint_{i}_{j}.pth')
    
            if os.path.exists(best_model_path):
                diffusion.load_state_dict(torch.load(best_model_path))
                logging.info(f"載入單元 ({i}, {j}) 的最佳模型: {best_model_path}")
            elif os.path.exists(checkpoint_model_path):
                checkpoint = torch.load(checkpoint_model_path)
                diffusion.load_state_dict(checkpoint['model_state_dict'])
                logging.warning(f"單元 ({i}, {j}) 的最佳模型不存在，載入檢查點模型: {checkpoint_model_path}")
            else:
                raise FileNotFoundError(f"單元 ({i}, {j}) 的模型文件不存在，請先訓練模型")
    
            models[(i, j)] = diffusion

    metrics = evaluate_model_per_cell(models, grid_data, test_dataset, device=device, max_samples=20)

    logging.info(f"重建 MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, MAPE: {metrics['mape']:.6f}, SMAPE: {metrics['smape']:.6f}")
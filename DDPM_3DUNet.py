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

# -------------------------------
# 設定 logging 等級，方便印出訓練與評估時的訊息
# -------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------
# 常數定義：時間嵌入維度
# -------------------------------
TIME_EMB_DIM = 128

# --------------------------------------
# 數據處理相關
# --------------------------------------
def parse_lat_lon(column_name: str) -> tuple[float, float]:
    """
    解析欄位名稱中的經緯度資訊，假設格式為 "name (lon, lat)"。

    參數:
        column_name: 欄位名稱，必須包含以括號包住的經緯度資訊，例如 "(121.565, 25.033)"。

    回傳:
        一個元組 (經度, 緯度) 的浮點數。

    若格式不正確則拋出 ValueError。
    """
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class PeopleFlowDatasetCondition(Dataset):
    """
    自訂 Dataset 類別，用來處理人流數據（CSV 格式），
    並將數據依據經緯度對應到固定網格中。

    參數:
        csv_path: CSV 檔案的路徑。
        H, W: 網格的高度與寬度。
        condition_length: 條件序列的長度（例如前幾個時間步）。
        prediction_length: 預測序列的長度（例如後續預測一個時間步）。
        transform: 可選的轉換函數，預設為 None。
        normalize: 是否對數據進行正規化處理，預設為 True。
        debug: 是否啟用除錯模式，若為 True 則會繪製網格圖。
    """
    def __init__(self, csv_path: str, H: int, W: int, condition_length: int, 
                 prediction_length: int, transform: Optional[callable] = None, 
                 normalize: bool = True, debug: bool = False):
        # 若 CSV 檔案不存在則拋出錯誤
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 檔案未找到：{csv_path}")
        
        # 讀取 CSV 數據
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.condition_length = condition_length
        self.prediction_length = prediction_length
        self.total_length = condition_length + prediction_length
        self.normalize = normalize
        self.H, self.W = H, W

        # 找出包含經緯度資訊的欄位（欄位名稱中包含括號）
        flow_columns = [c for c in self.df.columns if '(' in c and ')' in c]
        # 解析每個欄位名稱，取得 (欄位名稱, 經度, 緯度)
        column_info = [(col, *parse_lat_lon(col)) for col in flow_columns]
        # 將經緯度資料取出並組成 numpy 陣列，注意順序為 (lon, lat)
        coords = np.array([(lon, lat) for _, lon, lat in column_info])

        # --------------------------------------
        # 計算網格中心點：以所有座標的平均值作為中心
        # --------------------------------------
        mean_lon, mean_lat = np.mean(coords, axis=0)
        distances_to_center = np.sqrt((coords[:, 0] - mean_lon)**2 + (coords[:, 1] - mean_lat)**2)
        central_idx = np.argmin(distances_to_center)  # 與中心點最近的索引
        central_coord = coords[central_idx]

        # --------------------------------------
        # 初始化固定尺寸的網格 (21x21)，並將中心點填入中間位置 (10,10)
        # --------------------------------------
        grid_size = 21
        grid = np.full((grid_size, grid_size), -1, dtype=int)
        central_row, central_col = 10, 10
        grid[central_row, central_col] = central_idx

        # --------------------------------------
        # 計算網格步長：根據經緯度差距取中位數作為步長
        # --------------------------------------
        lon_diffs = np.diff(np.sort(coords[:, 0]))
        lat_diffs = np.diff(np.sort(coords[:, 1]))
        lon_step = np.median(lon_diffs[lon_diffs > 0]) if len(lon_diffs) > 0 else 0.005
        lat_step = np.median(lat_diffs[lat_diffs > 0]) if len(lat_diffs) > 0 else 0.005

        # --------------------------------------
        # 分配剩餘的座標到網格其他位置
        # --------------------------------------
        available_indices = list(range(len(coords)))
        available_indices.remove(central_idx)
        grid_positions = []
        # 以 k 為曼哈頓距離層級，從中心向外分配
        for k in range(11):
            for r in range(max(0, 10 - k), min(21, 10 + k + 1)):
                for c in range(max(0, 10 - k), min(21, 10 + k + 1)):
                    if max(abs(r - 10), abs(c - 10)) == k:
                        grid_positions.append((r, c))

        # 對每個網格位置依據目標座標與可用座標間的距離，選擇最接近的
        for r, c in grid_positions:
            if grid[r, c] != -1:
                continue  # 跳過已填入的中心點
            # 根據網格位置計算目標經緯度：經度根據列數調整，緯度則根據行數調整
            target_lon = central_coord[0] + (c - central_col) * lon_step
            target_lat = central_coord[1] - (r - central_row) * lat_step

            # 根據位置選擇適當的經緯度約束條件
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

            # 從剩餘可用的座標中篩選符合約束條件的索引
            filtered_indices = [idx for idx in available_indices if
                                (lon_constraint is None or lon_constraint(coords[idx][0])) and
                                (lat_constraint is None or lat_constraint(coords[idx][1]))]
            if filtered_indices:
                # 若有符合條件的座標，選取與目標距離最接近的那一個
                distances = np.sqrt((coords[filtered_indices, 0] - target_lon)**2 +
                                    (coords[filtered_indices, 1] - target_lat)**2)
                closest_idx = filtered_indices[np.argmin(distances)]
            else:
                # 若無符合條件的，則從全部剩餘中選取最接近的
                distances = np.sqrt((coords[available_indices, 0] - target_lon)**2 +
                                    (coords[available_indices, 1] - target_lat)**2)
                closest_idx = available_indices[np.argmin(distances)]
            
            # 將選取的座標填入網格中
            grid[r, c] = closest_idx
            available_indices.remove(closest_idx)

        # 確保整個網格皆已填滿，否則拋出錯誤
        if len(grid[grid != -1]) != grid_size * grid_size:
            raise ValueError(f"網格未填滿：選取 {len(grid[grid != -1])} 個，需 {grid_size * grid_size} 個")

        # 將網格中的索引轉換為對應的欄位名稱，依照網格展平成一維排列
        sorted_indices = grid.flatten()
        self.sorted_flow_columns = [column_info[idx][0] for idx in sorted_indices]
        # 若啟用除錯模式則繪製網格圖，存至指定路徑
        if debug:
            self._plot_grid(save_path=r"C:\thesis\code\result_ddpm\plot_grid.png")

        # 根據排序好的欄位順序取出 CSV 中的數據，並調整形狀為 (time, H, W)
        flow_values = self.df[self.sorted_flow_columns].values.reshape(-1, H, W).astype(np.float32)
        self.data = torch.from_numpy(flow_values)
        
        # 若啟用正規化，計算均值與標準差，並調整數據
        if normalize:
            self.mean_val = self.data.mean()
            self.std_val = self.data.std() + 1e-5
            self.data = (self.data - self.mean_val) / self.std_val
        
        # 計算數據集中可以取樣的最大索引
        self.max_index = self.data.shape[0] - self.total_length + 1

    def _plot_grid(self, save_path: str):
        """
        繪製網格圖，顯示每個網格點對應的經緯度及其座標位置，
        並將結果存檔到指定的路徑。
        """
        locations = [parse_lat_lon(col) for col in self.sorted_flow_columns]
        longitudes, latitudes = zip(*locations)
        plt.figure(figsize=(12, 12))
        plt.scatter(longitudes, latitudes, c='blue', marker='o', label='Grid Points')
        # 在每個點旁顯示網格索引
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

    def __len__(self) -> int:
        # 數據集的總樣本數為可滑動視窗的數量
        return self.max_index

    def __getitem__(self, idx):
        """
        根據給定的索引，返回模型輸入與目標序列。
        輸出:
            model_input: 包含條件序列與目標序列的合併結果，形狀 (1, condition_length+prediction_length, H, W)
            target_seq: 真實的目標序列，形狀 (1, 1, H, W)
        """
        cond_seq = self.data[idx:idx + self.condition_length]  # (8, 21, 21)
        target_seq = self.data[idx + self.condition_length:idx + self.total_length]  # (1, 21, 21)
        # 將條件序列與目標序列沿著時間維度連接，並加上 batch 維度
        model_input = torch.cat([cond_seq, target_seq], dim=0).unsqueeze(0)  # (9, 21, 21) -> (1, 9, 21, 21)
        return model_input, target_seq.unsqueeze(0)  # 返回 (1, 9, 21, 21) 與 (1, 1, 21, 21)

def collate_fn(batch):
    """
    自訂批次處理函數，用於 DataLoader。
    將每個樣本的 model_input 與 target 分別堆疊成 batch。
    """
    conds, targets = zip(*batch)
    return torch.stack(conds), torch.stack(targets)

# --------------------------------------
# 模型定義
# --------------------------------------
class DoubleConv3D(nn.Module):
    """
    定義 3D 卷積層組合，包含兩次卷積、BatchNorm 與 ReLU 激活函數。
    此結構常用於 U-Net 中作為基本模組。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一次卷積
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # 第二次卷積
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net 結構，包含下採樣（Encoder）、中間瓶頸層與上採樣（Decoder）。
    此模型同時接收噪聲版本的目標數據 x_t、完整序列 x_full 與時間嵌入。
    """
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, dropout_rate=0.0):
        super().__init__()
        # 編碼器部分：逐層進行雙卷積與下採樣
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        # 這裡使用不同的池化參數以調整深度與空間尺寸
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        # 瓶頸層
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)
        # 解碼器部分：逐層上採樣並與對應編碼層做 concat
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=(2, 2, 2), stride=(1, 2, 2), output_padding=(0, 1, 1))
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2), output_padding=(1, 0, 0))
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        # 輸出卷積，將通道數降為 1
        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
        # dropout 用於防止過擬合
        self.dropout = nn.Dropout3d(dropout_rate)
        # 時間嵌入的線性轉換與激活
        self.time_proj = nn.Sequential(nn.Linear(time_emb_dim, base_channels * 8), nn.SiLU())
        # 將完整序列 x_full 通過 1x1 卷積調整通道數，使其與 x_t 保持一致（這裡假設保持 1 通道）
        self.x_full_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_t, x_full, t_emb):
        """
        前向傳播函數：
        參數:
            x_t: 含噪聲的部分序列，形狀 (batch, 1, 9, 21, 21)
            x_full: 完整的序列數據，形狀 (batch, 1, 9, 21, 21)
            t_emb: 時間嵌入，形狀 (batch, time_emb_dim)
        回傳:
            模型輸出，形狀 (batch, 1, 1, 21, 21)
        """
        # 處理 x_full，使其通道數與 x_t 一致
        x_full_conv = self.x_full_conv(x_full)  # (batch, 1, 9, 21, 21)
        # 將 x_t 與處理後的 x_full 做融合（逐元素相加）
        x_input = x_t + x_full_conv  # (batch, 1, 9, 21, 21)
        # 編碼器第一層
        e1 = self.enc1(x_input)  # (batch, 64, 9, 21, 21)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        p4 = self.pool4(e4)
        # 將時間嵌入經線性轉換後擴展至與 p4 同維度並與 p4 相加
        t_emb = self.time_proj(t_emb)[:, :, None, None, None]
        b = self.bottleneck(p4 + t_emb)
        b = self.dropout(b)
        # 解碼器：上採樣後與對應編碼層做 concat
        d4 = self.up4(b)
        if d4.shape[-3:] != e4.shape[-3:]:
            # 使用 trilinear 插值調整尺寸
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
        # 經過輸出卷積獲得最終結果
        out = self.out_conv(d1)
        # 返回結果，僅保留時間維度上的第一個步驟（1個預測步長）
        return out[:, :, :1, :, :]

class DDPM3D(nn.Module):
    """
    條件式 DDPM (Denoising Diffusion Probabilistic Model) 模型。
    此模型利用前向擴散與反向去噪過程進行生成任務。
    """
    def __init__(self, model: nn.Module, timesteps: int = 1000, 
                 beta_start: float = 1e-4, beta_end: float = 0.02, device: str = 'cuda'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        # 線性生成 beta 值
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        # 累乘計算 alpha 的連乘積，用於生成擴散過程的係數
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # 計算頻率因子，用於時間嵌入的正弦與餘弦函數
        self.half_dim = TIME_EMB_DIM // 2
        self.freq_factor = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) *
                                     -(math.log(10000.0) / (self.half_dim - 1))).to(device)

    def get_time_embedding(self, t):
        """
        生成時間嵌入向量，使用正弦與餘弦函數將標量時間映射到向量。
        參數:
            t: 時間步（batch_size,)
        回傳:
            時間嵌入，形狀 (batch_size, TIME_EMB_DIM)
        """
        t = t.float()
        emb = t[:, None] * self.freq_factor.to(t.device)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def get_condition_embedding(self, cond):
        """
        取得條件嵌入，目前未使用條件資訊，返回全零向量。
        """
        return torch.zeros(cond.shape[0], TIME_EMB_DIM, device=cond.device)

    def q_sample(self, x0, t, noise=None):
        """
        前向擴散過程：根據給定的時間步 t，將數據 x0 擴散成含噪版本。
        參數:
            x0: 原始數據
            t: 時間步（batch_size,)
            noise: 可選噪聲，若未提供則生成隨機噪聲
        回傳:
            擴散後的數據
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # 根據時間步獲取相對應的係數
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        # 混合原始數據與噪聲
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def p_losses(self, cond, target, t):
        """
        計算去噪損失，目標是讓模型預測出噪聲部分。
        參數:
            cond: 條件序列（含目標數據與前置條件）
            target: 真實目標數據
            t: 時間步
        回傳:
            均方誤差損失
        """
        # 構造完整無噪聲序列：將 target 與條件序列的後半部拼接
        x_full = torch.cat([target, cond[:, :, 1:]], dim=2)
        noise = torch.randn_like(target)
        # 生成含噪 target
        x_noisy_target = self.q_sample(target, t, noise=noise)
        # 將含噪 target 與條件序列拼接，作為模型輸入
        x_t = torch.cat([x_noisy_target, cond[:, :, 1:]], dim=2)
        # 取得時間嵌入與條件嵌入，並融合
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        # 預測噪聲
        pred_noise = self.model(x_t, x_full, combined_emb)
        pred_noise_target = pred_noise[:, :, :1, :, :]
        # 計算模型預測與實際噪聲間的均方誤差
        return F.mse_loss(pred_noise_target, noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        """
        單步反向去噪：從含噪數據 x_t 生成前一步的數據。
        參數:
            x_t: 當前含噪數據，形狀 (batch, 1, 1, H, W) 或 (batch, 1, 9, 21, 21)
            t: 當前時間步
            cond: 條件數據
        回傳:
            去噪後的數據 x_{t-1}
        """
        # 確保 x_t 與 cond 為 5 維張量
        if x_t.dim() == 4:
            x_t = x_t.unsqueeze(1)  # (batch, 1, 1, H, W)
        if cond.dim() == 4:
            cond = cond.unsqueeze(1)  # (batch, 1, 9, 21, 21)
        
        # 取得當前時間步的 beta 等係數
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        
        # 將含噪數據與條件數據合併
        x_t_full = torch.cat([x_t, cond[:, :, 1:]], dim=2)  # (batch, 1, 9, 21, 21)
        x_full = cond  # 完整序列作為條件
        # 模型預測噪聲
        eps_theta = self.model(x_t_full, x_full, combined_emb)
        eps_theta_target = eps_theta[:, :, :1, :, :]  # 僅取出目標噪聲部分
        
        # 反向去噪步驟：計算 x_{t-1}
        x_t_minus_1 = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta_target)
        # 當 t > 0 時，加入隨機噪聲；t = 0 時直接返回
        mask = (t > 0).float().view(-1, 1, 1, 1, 1)
        sigma_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        return x_t_minus_1 + mask * sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        反向去噪迴圈：從初始純噪聲開始，逐步去噪生成數據。
        參數:
            shape: 生成數據的形狀（可能需要調整為 5D）
            cond: 條件數據
        回傳:
            生成的數據張量
        """
        # 確保 cond 為 5D
        if cond.dim() == 4:
            cond = cond.unsqueeze(1)  # (batch, 1, 9, 21, 21)
        
        # 調整 shape 為 5D
        if len(shape) == 4:
            batch_size = shape[0]
            shape = (batch_size, 1, shape[1], shape[2], shape[3])
        
        # 初始噪聲
        x = torch.randn(shape, device=self.device)
        
        # 由最後一步開始，逐步進行去噪
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        
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
                         save_dir: str = r"C:\thesis\code\result_ddpm"):
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
    
    # 對預測的每個時間步進行繪圖
    for t in range(pred_length):
        plt.figure(figsize=(16, 4))
        
        # 子圖1：生成結果
        plt.subplot(1, 4, 1)
        plt.imshow(generated[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Generated (t={t})')
        
        # 子圖2：真實值
        plt.subplot(1, 4, 2)
        plt.imshow(target[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'True (t={t})')
        
        # 子圖3：MSE 誤差圖（平方誤差）
        error_sq = (generated[sample_idx, 0, t].cpu().numpy() - target[sample_idx, 0, t].cpu().numpy()) ** 2
        plt.subplot(1, 4, 3)
        plt.imshow(error_sq, cmap='hot')
        plt.colorbar()
        plt.title(f'MSE (t={t})')
        
        # 子圖4：MAE 誤差圖（絕對誤差）
        error_abs = np.abs(generated[sample_idx, 0, t].cpu().numpy() - target[sample_idx, 0, t].cpu().numpy())
        plt.subplot(1, 4, 4)
        plt.imshow(error_abs, cmap='hot')
        plt.colorbar()
        plt.title(f'MAE (t={t})')
        
        plt.suptitle(f'Sample {sample_idx} - Time Step {t}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'prediction_sample{sample_idx}_t{t}.png'), dpi=300)
        plt.close()

def plot_grid_with_error(sorted_flow_columns: list, H: int, W: int, 
                         mse_matrix: np.ndarray, mae_matrix: np.ndarray, mape_matrix: np.ndarray, 
                         save_dir: str = r"C:\\thesis\\code\\result_ddpm"):
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
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 解析經緯度
    locations = [parse_lat_lon(col) for col in sorted_flow_columns]
    longitudes, latitudes = zip(*locations)
    
    # 定義顏色映射
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
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mae.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 繪製 MAPE 網格圖
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mape_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAPE (%)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAPE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mape.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
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
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'mse_mae_mape_per_coordinate.csv'), index=False)
    df.to_excel(os.path.join(save_dir, 'mse_mae_mape_per_coordinate.xlsx'), index=False)

# --------------------------------------
# 訓練與評估函數
# --------------------------------------
def train_ddpm(diffusion: DDPM3D, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 20, lr: float = 1e-4, device: str = 'cuda', 
               patience: int = 3, weight_decay: float = 1e-6, 
               save_dir: str = r"C:\thesis\code\result_ddpm") -> DDPM3D:
    """
    訓練 DDPM 模型，並進行驗證與早停檢查。
    參數:
        diffusion: DDPM 模型實例
        train_loader, val_loader: 訓練與驗證的 DataLoader
        epochs: 最大訓練輪數
        lr: 學習率
        device: 訓練設備，例如 'cuda' 或 'cpu'
        patience: 早停耐心次數
        weight_decay: 優化器的權重衰減
        save_dir: 模型與結果的存檔目錄
    回傳:
        訓練後的 diffusion 模型
    """
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    diffusion.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        diffusion.train()
        total_train_loss = 0
        # 逐批訓練
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            optimizer.zero_grad()
            # 隨機抽取一個時間步
            t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
            loss = diffusion.p_losses(cond, target, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 驗證模式
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for cond, target in val_loader:
                cond, target = cond.to(device), target.to(device)
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                loss = diffusion.p_losses(cond, target, t)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 若驗證損失降低則儲存模型，否則耐心計數增加
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(diffusion.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    # 繪製訓練與驗證損失曲線，並存檔
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

    return diffusion

@torch.no_grad()
def evaluate_model(diffusion: DDPM3D, dataset: Dataset, device: str = 'cuda', 
                   max_samples: int = 100, save_dir: str = r"C:\\thesis\\code\\result_ddpm",
                   sample_idx: int = 0) -> dict:
    """
    評估模型，並生成視覺化圖表。
    
    Args:
        diffusion: 訓練好的 DDPM 模型
        dataset: 驗證或測試數據集
        device: 設備
        max_samples: 最大評估樣本數
        save_dir: 結果存檔目錄
        sample_idx: 指定視覺化的樣本索引
    Returns:
        包含 MSE、MAE 和 MAPE 的評估指標字典
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import os
    import random
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from torch.utils.data import Subset

    diffusion.eval()
    metrics = {'mse': 0.0, 'mae': 0.0, 'mape': 0.0}
    N = min(len(dataset), max_samples)
    sample_indices = random.sample(range(len(dataset)), N)
    
    # 若 dataset 為 Subset，則取出原始數據集以獲取參數
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    H, W = base_dataset.H, base_dataset.W
    pred_length = base_dataset.prediction_length
    
    mean_val = base_dataset.mean_val.to(device)
    std_val = base_dataset.std_val.to(device)
    
    # 預先分配張量
    generated_batch = torch.zeros(N, 1, pred_length, H, W, device=device)
    target_batch = torch.zeros(N, 1, pred_length, H, W, device=device)
    
    for i, idx in enumerate(sample_indices):
        cond, target = dataset[idx]
        cond, target = cond.to(device), target.to(device)
        target = target.unsqueeze(2)  # (1, 1, 1, H, W)
        
        x_recon = diffusion.p_sample_loop(target.shape, cond)
        
        # 反正規化
        x_recon_original = x_recon * std_val + mean_val
        target_original = target * std_val + mean_val
        
        generated_batch[i] = x_recon_original
        target_batch[i] = target_original
        
        # 計算誤差
        mse = F.mse_loss(x_recon_original, target_original).item()
        mae = F.l1_loss(x_recon_original, target_original).item()
        mape = torch.mean(torch.abs((target_original - x_recon_original) / (target_original + 1e-10))) * 100
        
        metrics['mse'] += mse
        metrics['mae'] += mae
        metrics['mape'] += mape.item()
    
    # 平均誤差
    metrics['mse'] /= N
    metrics['mae'] /= N
    metrics['mape'] /= N
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 計算每個網格點的誤差矩陣
    error_matrix_mse = (generated_batch - target_batch) ** 2
    mse_matrix = torch.mean(error_matrix_mse, dim=(0, 2)).cpu().numpy()[0]  # (H, W)
    
    error_matrix_mae = torch.abs(generated_batch - target_batch)
    mae_matrix = torch.mean(error_matrix_mae, dim=(0, 2)).cpu().numpy()[0]  # (H, W)
    
    # 計算 MAPE 矩陣
    mape_matrix = torch.mean(torch.abs((target_batch - generated_batch) / (target_batch + 1e-10)), 
                            dim=(0, 2)).cpu().numpy()[0] * 100  # (H, W)
    
    # 繪製誤差圖並保存表格
    plot_grid_with_error(base_dataset.sorted_flow_columns, H, W, mse_matrix, mae_matrix, mape_matrix, save_dir)
    
    # 儲存評估結果
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Evaluation Metrics (computed on {N} samples):\n")
        f.write(f"Reconstruction MSE: {metrics['mse']:.6f}\n")
        f.write(f"Reconstruction MAE: {metrics['mae']:.6f}\n")
        f.write(f"Reconstruction MAPE: {metrics['mape']:.6f}%\n")
    
    return metrics

# --------------------------------------
# 主程式進入點
# --------------------------------------
if __name__ == "__main__":
    # -------------------------------
    # 參數設定：網格尺寸、序列長度、批次大小、訓練輪數等
    # -------------------------------
    H, W = 21, 21
    condition_length, prediction_length = 8, 1
    batch_size, epochs, lr, timesteps, patience = 100, 1000, 0.0001, 1000, 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = r"C:\thesis\code\result_ddpm"

    # 設定隨機種子，確保實驗結果可重現
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------
    # 初始化數據集，並依比例劃分訓練、驗證與測試集
    # -------------------------------
    dataset = PeopleFlowDatasetCondition(
        csv_path=r"C:\thesis\code\Taipei_CF\all_merged.csv",
        H=H, W=W, condition_length=condition_length, prediction_length=prediction_length,
        normalize=True, debug=True
    )
    train_end = int(0.7 * len(dataset))
    val_end = int(0.85 * len(dataset))
    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))
    test_dataset = Subset(dataset, range(val_end, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # -------------------------------
    # 初始化模型
    # 使用 UNet3D 作為去噪模型，並建立 DDPM3D 實例
    # -------------------------------
    unet = UNet3D(in_channels=1, base_channels=64, time_emb_dim=TIME_EMB_DIM, dropout_rate=0.0)
    diffusion = DDPM3D(model=unet, timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)

    # -------------------------------
    # 訓練模型
    # -------------------------------
    trained_diffusion = train_ddpm(diffusion, train_loader, val_loader, epochs=epochs, 
                                    lr=lr, device=device, patience=patience, save_dir=save_dir)

    # -------------------------------
    # 評估模型
    # -------------------------------
    metrics = evaluate_model(trained_diffusion, val_dataset, device=device, max_samples=100, save_dir=save_dir)
    logging.info(f"Reconstruction MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")

    # -------------------------------
    # 儲存最終評估結果
    # -------------------------------
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Evaluation Metrics (computed on 2 samples):\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Reconstruction MSE: {metrics['mse']:.6f}\n")
        f.write(f"Reconstruction MAE: {metrics['mae']:.6f}\n")
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump({
            "mse": metrics['mse'], "mae": metrics['mae'],
            "sample_size": 2, "timestamp": pd.Timestamp.now().isoformat()
        }, f, indent=4)

#%%
# -*- coding: utf-8 -*-
import os
import random
import re
import math
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional

# ==============================================================================
# 腳本說明
# ==============================================================================
# 功能:
# 1. 載入預訓練的 Basemodel, Stage2, Stage3 模型。
# 2. 對完整的 16982 小時資料集進行模擬預測。
# 3. 計算每個網格的累計正規化誤差: Sum((模擬值 - 實際值) / 網格STD)。
# 4. 視覺化誤差分佈地圖並存檔。
# 5. 將網格依誤差分為四組 (正高/正低/負高/負低)，視覺化分組地圖並存檔。
# 6. 將網格座標、累計誤差、分組結果匯出至 Excel。
#
# 使用方式:
# 1. 確認 CONFIG 中的所有檔案路徑皆正確。
# 2. 執行 `python analyze_stage3_error.py`。
# 3. 結果將儲存在 `output_dir` 指定的目錄中。
# ==============================================================================

# ==============================================================================
# 組態設定
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    # --- 路徑設定 ---
    "data_path": r"C:\thesis\code\Taipei_CF\all_merged.csv",
    "basemodel_checkpoint": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_long-term\best_ddpm_model_during_training.pth",
    "stage2_checkpoint": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage2\Stage2_HourLe20\best_stage2_model_hour_le_20.pth",
    "stage3_checkpoint": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage3\Stage3_WeekdayLe4\best_stage3_model_Weekday_le_4.pth",
    "cache_dir":  r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage3\Stage3_WeekdayLe4\analysis_cache",
    "output_dir": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage3\Stage3_WeekdayLe4\analysis_error", # 所有輸出 (圖片, Excel) 的儲存目錄

    # --- 資料與模型通用參數 (必須與訓練時一致) ---
    "H": 20,
    "W": 20,
    "D": 1,
    "image_channels": 1,
    "base_channels_unet": 64,
    "time_emb_dim": 256,
    "condition_encode_dim": 16,
    "condition_input_channels": 2, # Stage2/3 使用2個網格作為條件
    "timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 推論參數 ---
    "batch_size": 256, # 可依據您的 GPU 記憶體調整
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

# 建立輸出目錄
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["cache_dir"], exist_ok=True) 
CACHE_FULL_PATH = CONFIG["cache_dir"]
logger.info(f"所有分析結果將儲存於: {CONFIG['output_dir']}")
logger.info(f"模型推論快取將儲存於: {CACHE_FULL_PATH}") 

# 設定隨機種子以確保可重現性
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])

# ==============================================================================
# 模型定義 (從 DDPM_Long-term_3stage.py 複製)
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device; half_dim = self.dim // 2; emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]; emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, kernel_size: int = 3, padding: int = 1):
        super().__init__(); mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm3d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm3d(out_channels), nn.SiLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.double_conv(x)

class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), DoubleConv3D(in_channels, out_channels))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.maxpool_conv(x)

class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__(); self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv3D(in_channels, out_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[3] - x1.size()[3]; diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int): super().__init__(); self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, input_image_channels: int, base_channels: int = 64, time_emb_dim: int = 256,
                 condition_encode_dim: Optional[int] = None, bilinear_upsample: bool = True, dropout_rate: float = 0.0):
        super().__init__()
        self.input_image_channels = input_image_channels; self.condition_encode_dim = condition_encode_dim or 0
        self.shared_time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        actual_in_channels = self.input_image_channels + self.condition_encode_dim
        self.inc = DoubleConv3D(actual_in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2); self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8); factor = 2 if bilinear_upsample else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor); self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, bilinear_upsample); self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, bilinear_upsample)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, bilinear_upsample); self.up4 = Up3D(base_channels * 2, base_channels, bilinear_upsample)
        self.outc = OutConv3D(base_channels, self.input_image_channels)
        self.time_proj_inc = nn.Linear(time_emb_dim, base_channels); self.time_proj_down1 = nn.Linear(time_emb_dim, base_channels * 2)
        self.time_proj_down2 = nn.Linear(time_emb_dim, base_channels * 4); self.time_proj_down3 = nn.Linear(time_emb_dim, base_channels * 8)
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, base_channels * 16 // factor); self.time_proj_up1 = nn.Linear(time_emb_dim, base_channels * 8 // factor)
        self.time_proj_up2 = nn.Linear(time_emb_dim, base_channels * 4 // factor); self.time_proj_up3 = nn.Linear(time_emb_dim, base_channels * 2 // factor)
        self.time_proj_up4 = nn.Linear(time_emb_dim, base_channels)
    def _add_time_embedding(self, x: torch.Tensor, t_emb_projected: torch.Tensor) -> torch.Tensor:
        return x + t_emb_projected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    def forward(self, x_t: torch.Tensor, time_steps: torch.Tensor, processed_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        shared_t_emb = self.shared_time_mlp(time_steps)
        x_input = torch.cat((x_t, processed_condition), dim=1) if processed_condition is not None else x_t
        x1 = self._add_time_embedding(self.inc(x_input), self.time_proj_inc(shared_t_emb))
        x2 = self._add_time_embedding(self.down1(x1), self.time_proj_down1(shared_t_emb))
        x3 = self._add_time_embedding(self.down2(x2), self.time_proj_down2(shared_t_emb))
        x4 = self._add_time_embedding(self.down3(x3), self.time_proj_down3(shared_t_emb))
        x5 = self.dropout(self._add_time_embedding(self.down4(x4), self.time_proj_bottleneck(shared_t_emb)))
        x = self._add_time_embedding(self.up1(x5, x4), self.time_proj_up1(shared_t_emb))
        x = self._add_time_embedding(self.up2(x, x3), self.time_proj_up2(shared_t_emb))
        x = self._add_time_embedding(self.up3(x, x2), self.time_proj_up3(shared_t_emb))
        x = self._add_time_embedding(self.up4(x, x1), self.time_proj_up4(shared_t_emb))
        return self.outc(x)

def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM3D(nn.Module):
    def __init__(self, unet_model: UNet3D, timesteps: int, image_size: Tuple[int, int, int], image_channels: int,
                 condition_input_channels: int, condition_encode_dim: int, beta_start: float, beta_end: float, device: str):
        super().__init__()
        self.model = unet_model; self.timesteps = timesteps; self.image_size_D, self.image_size_H, self.image_size_W = image_size
        self.image_channels = image_channels; self.device = device
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas; self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.condition_processor = nn.Sequential(
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()).to(device)
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]; out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    def _prepare_original_conditional_input_grids(self, hour_scalars_batch: torch.Tensor, is_holiday_scalars_batch: torch.Tensor) -> torch.Tensor:
        batch_size = hour_scalars_batch.shape[0]; norm_hours = hour_scalars_batch.float().to(self.device) / 23.0
        holiday_values = is_holiday_scalars_batch.float().to(self.device)
        hour_grid_vals = norm_hours.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)
        holiday_grid_vals = holiday_values.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)
        hour_grids_t = hour_grid_vals.unsqueeze(1).unsqueeze(2).repeat(1,1,self.image_size_D,1,1)
        holiday_grids_t = holiday_grid_vals.unsqueeze(1).unsqueeze(2).repeat(1,1,self.image_size_D,1,1)
        return torch.cat((hour_grids_t, holiday_grids_t), dim=1).to(self.device)
    def _prepare_stage2_or_3_condition_grids(self, cond1_grid: torch.Tensor, cond2_grid: torch.Tensor) -> torch.Tensor:
        return torch.cat((cond1_grid, cond2_grid), dim=1)
    @torch.no_grad()
    def sample(self, batch_size: int, **kwargs) -> torch.Tensor:
        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)
        processed_conditions = None
        if "hour_scalars_batch" in kwargs and "is_holiday_scalars_batch" in kwargs:
            stacked_cond_grids = self._prepare_original_conditional_input_grids(kwargs["hour_scalars_batch"], kwargs["is_holiday_scalars_batch"])
            processed_conditions = self.condition_processor(stacked_cond_grids)
        elif "cond1_grid" in kwargs and "cond2_grid" in kwargs:
            stacked_cond_grids = self._prepare_stage2_or_3_condition_grids(kwargs["cond1_grid"], kwargs["cond2_grid"])
            processed_conditions = self.condition_processor(stacked_cond_grids)
        else: raise ValueError("Sample method requires valid conditions.")
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            betas_t = self._extract(self.betas, t, img.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, img.shape)
            predicted_noise = self.model(img, t, processed_conditions)
            model_mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            if i == 0: img = model_mean
            else:
                posterior_variance_t = self._extract(self.posterior_variance, t, img.shape)
                noise = torch.randn_like(img); img = model_mean + torch.sqrt(posterior_variance_t) * noise
        return img
# ==============================================================================
# 輔助函式 (模型載入、繪圖)
# ==============================================================================
def load_ddpm_model(checkpoint_path: str, config: Dict[str, Any], device: str) -> Tuple[DDPM3D, Dict[str, Any]]:
    """從檢查點載入一個 DDPM3D 模型及其相關配置。"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"檢查點檔案不存在: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    chkpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = chkpt.get('config_snapshot_at_save', config)

    unet = UNet3D(
        input_image_channels=model_config.get("image_channels", config["image_channels"]),
        base_channels=model_config.get("base_channels_unet", config["base_channels_unet"]),
        time_emb_dim=model_config.get("time_emb_dim", config["time_emb_dim"]),
        condition_encode_dim=model_config.get("condition_encode_dim", config["condition_encode_dim"]),
    ).to(device)

    # Basemodel 可能有不同的條件輸入通道數
    cond_input_ch = model_config.get("condition_input_channels", config["condition_input_channels"])

    ddpm_model = DDPM3D(
        unet_model=unet,
        timesteps=model_config.get("timesteps", config["timesteps"]),
        image_size=(model_config.get("D", config["D"]), model_config.get("H", config["H"]), model_config.get("W", config["W"])),
        image_channels=model_config.get("image_channels", config["image_channels"]),
        condition_input_channels=cond_input_ch,
        condition_encode_dim=model_config.get("condition_encode_dim", config["condition_encode_dim"]),
        beta_start=model_config.get("beta_start", config["beta_start"]),
        beta_end=model_config.get("beta_end", config["beta_end"]),
        device=device
    )

    ddpm_model.load_state_dict(chkpt['ddpm_state_dict'])
    ddpm_model.eval()
    logger.info(f"成功從 {os.path.basename(checkpoint_path)} 載入模型 (Epoch {chkpt.get('epoch', 'N/A')})。")
    return ddpm_model, chkpt

def plot_grid_map(
    grid_values: np.ndarray,
    title: str,
    output_filename: str,
    config: Dict[str, Any],
    grid_map_info: Dict[str, Any],
    cmap: Any = 'viridis',
    is_categorical: bool = False,
    cat_labels: Optional[List[str]] = None,
    cbar_label: str = "Value",
    vmin: Optional[float] = None,  # <--- 新增 vmin 參數
    vmax: Optional[float] = None   # <--- 新增 vmax 參數
):
    """繪製地理網格分佈圖並儲存。"""
    H, W = config["H"], config["W"]
    save_path = os.path.join(config["output_dir"], output_filename)

    sorted_flow_columns = grid_map_info.get("sorted_flow_columns")
    selected_sensor_info = grid_map_info.get("selected_sensor_info")

    if not sorted_flow_columns or not selected_sensor_info:
        logger.error("缺少繪圖所需的網格映射資訊 (sorted_flow_columns or selected_sensor_info)。")
        return

    sensor_coords = {info['name']: (info['lon'], info['lat']) for info in selected_sensor_info}
    
    lons, lats, plot_values = [], [], []
    grid_values_flat = grid_values.flatten()

    for i, col_name in enumerate(sorted_flow_columns):
        if i < len(grid_values_flat) and col_name in sensor_coords:
            lon, lat = sensor_coords[col_name]
            lons.append(lon)
            lats.append(lat)
            plot_values.append(grid_values_flat[i])

    if not lons:
        logger.error("沒有有效的感測器座標可供繪製。")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if is_categorical and cat_labels:
        n_cats = len(cat_labels)
        if isinstance(cmap, list):
            if len(cmap) < n_cats:
                raise ValueError(f"提供的顏色列表有 {len(cmap)} 種顏色，但需要 {n_cats} 種。")
            cmap_cat = mcolors.ListedColormap(cmap)
        else:
            cmap_cat = plt.get_cmap(cmap, n_cats)
            
        scatter = ax.scatter(lons, lats, c=plot_values, cmap=cmap_cat, marker='s', s=120, vmin=-0.5, vmax=n_cats-0.5)
        cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_cats))
        cbar.set_ticklabels(cat_labels)
    else:
        # --- 【核心修改】使用傳入的 vmin, vmax ---
        scatter = ax.scatter(lons, lats, c=plot_values, cmap=cmap, marker='s', s=120, vmin=vmin, vmax=vmax)
        fig.colorbar(scatter, ax=ax, label=cbar_label)

    ax.set_xlabel("經度 (Longitude)")
    ax.set_ylabel("緯度 (Latitude)")
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"已儲存地圖: {save_path}")
#%%
# ==============================================================================
# 主程式
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 載入模型 ---
    logger.info("===== 步驟 1: 載入所有預訓練模型 =====")
    device = CONFIG['device']
    basemodel, basemodel_chkpt = load_ddpm_model(CONFIG['basemodel_checkpoint'], CONFIG, device)
    stage2_model, stage2_chkpt = load_ddpm_model(CONFIG['stage2_checkpoint'], CONFIG, device)
    stage3_model, stage3_chkpt = load_ddpm_model(CONFIG['stage3_checkpoint'], CONFIG, device)

    # --- 2. 獲取必要的正規化與地理資訊 ---
    logger.info("===== 步驟 2: 提取正規化與地理資訊 =====")
    # Basemodel 提供地理資訊與原始流量正規化參數
    grid_map_info = {
        "sorted_flow_columns": basemodel_chkpt.get("sorted_flow_columns"),
        "selected_sensor_info": basemodel_chkpt.get("selected_sensor_info"),
        "grid_idx_to_rc_map": basemodel_chkpt.get("grid_idx_to_rc_map")
    }
    bm_norm_stats = basemodel_chkpt.get('norm_stats_flow', {'mean': 0, 'std': 1})
    
    # Stage2/3 提供其條件與目標的正規化參數
    s2_cond_norm_stats = stage2_chkpt.get('new_cond_feature_norm_stats')
    s2_target_norm_stats = stage2_chkpt.get('norm_stats_stage2_target')
    s3_cond_norm_stats = stage3_chkpt.get('s3_new_cond_feature_norm_stats')
    s3_target_norm_stats = stage3_chkpt.get('norm_stats_stage3_target')

    if any(v is None for v in [grid_map_info["sorted_flow_columns"], s2_cond_norm_stats, s3_cond_norm_stats, s3_target_norm_stats]):
        raise ValueError("一個或多個模型檢查點中缺少必要的正規化或地理資訊。")

    # --- 3. 載入並準備資料 ---
    logger.info("===== 步驟 3: 載入並準備完整資料集 =====")
    df = pd.read_csv(CONFIG['data_path'])
    df.rename(columns={'hoilday': 'holiday'}, inplace=True)
    logger.info(f"資料集 '{CONFIG['data_path']}' 載入完成，共 {len(df)} 筆資料。")

    flow_columns = grid_map_info["sorted_flow_columns"]
    actual_flow_data = df[flow_columns].values.reshape(len(df), CONFIG['H'], CONFIG['W'])

    # --- 4. 計算每個網格的實際人流標準差 ---
    logger.info("===== 步驟 4: 計算各網格流量的標準差 =====")
    grid_stds = np.std(actual_flow_data, axis=0)
    grid_stds[grid_stds == 0] = 1.0 # 避免除以零
    logger.info(f"標準差計算完成。形狀: {grid_stds.shape}")
#%%
    # --- 5. 進行模型推論並收集結果 ---
    logger.info("===== 步驟 5: 執行三階段模型推論 =====")
    # --- Stage 1: Basemodel 推論 ---
    bm_output_cache_path = os.path.join(CACHE_FULL_PATH, "basemodel_outputs_denorm.npy")
    if os.path.exists(bm_output_cache_path):
        logger.info(f"從快取載入 Basemodel 輸出: {bm_output_cache_path}")
        basemodel_outputs_denorm = np.load(bm_output_cache_path)
    else:
        logger.info("未找到 Basemodel 快取，開始執行推論...")
        all_bm_outputs = []
        pbar_bm = tqdm(range(0, len(df), CONFIG['batch_size']), desc="Basemodel 推論中")
        for i in pbar_bm:
            batch_df = df.iloc[i : i + CONFIG['batch_size']]
            hours_bm = torch.tensor(batch_df['時'].values, dtype=torch.long).to(device)
            holidays_bm = torch.tensor(batch_df['holiday'].values, dtype=torch.long).to(device)
            with torch.no_grad():
                bm_output_norm = basemodel.sample(batch_size=len(batch_df), hour_scalars_batch=hours_bm, is_holiday_scalars_batch=holidays_bm)
            bm_output_denorm = bm_output_norm * bm_norm_stats['std'] + bm_norm_stats['mean']
            all_bm_outputs.append(bm_output_denorm.cpu().numpy())
        basemodel_outputs_denorm = np.concatenate(all_bm_outputs, axis=0)
        np.save(bm_output_cache_path, basemodel_outputs_denorm)
        logger.info(f"Basemodel 推論完成並已儲存快取: {bm_output_cache_path}")

    # --- Stage 2: Stage2 推論 ---
    s2_output_cache_path = os.path.join(CACHE_FULL_PATH, "stage2_outputs_norm.npy")
    if os.path.exists(s2_output_cache_path):
        logger.info(f"從快取載入 Stage2 輸出: {s2_output_cache_path}")
        stage2_outputs_norm = np.load(s2_output_cache_path)
    else:
        logger.info("未找到 Stage2 快取，開始執行推論...")
        # 將 denorm 的 BM 輸出重新正規化為 S2 模型的條件 (使用BM的統計量)
        s2_cond1_grid_full = (basemodel_outputs_denorm - bm_norm_stats['mean']) / bm_norm_stats['std']
        all_s2_outputs = []
        pbar_s2 = tqdm(range(0, len(df), CONFIG['batch_size']), desc="Stage2 推論中")
        for i in pbar_s2:
            batch_df = df.iloc[i : i + CONFIG['batch_size']]
            s2_cond_values = torch.tensor(batch_df['時'].values, dtype=torch.float32).to(device)
            s2_cond_norm = (s2_cond_values - s2_cond_norm_stats['mean']) / s2_cond_norm_stats['std']
            s2_cond2_grid = s2_cond_norm.view(-1, 1, 1, 1, 1).expand(-1, 1, CONFIG['D'], CONFIG['H'], CONFIG['W'])
            
            s2_cond1_batch = torch.from_numpy(s2_cond1_grid_full[i : i + len(batch_df)]).to(device)

            with torch.no_grad():
                s2_output_norm = stage2_model.sample(batch_size=len(batch_df), cond1_grid=s2_cond1_batch, cond2_grid=s2_cond2_grid)
            all_s2_outputs.append(s2_output_norm.cpu().numpy())
        stage2_outputs_norm = np.concatenate(all_s2_outputs, axis=0)
        np.save(s2_output_cache_path, stage2_outputs_norm)
        logger.info(f"Stage2 推論完成並已儲存快取: {s2_output_cache_path}")
#%%
    # --- Stage 3: Stage3 推論 (最終模擬) ---
    s3_output_cache_path = os.path.join(CACHE_FULL_PATH, "stage3_outputs_denorm.npy")
    if os.path.exists(s3_output_cache_path):
        logger.info(f"從快取載入最終模擬結果: {s3_output_cache_path}")
        simulated_flow_data = np.load(s3_output_cache_path)
    else:
        logger.info("未找到最終模擬結果快取，開始執行 Stage3 推論...")
        all_s3_outputs = []
        pbar_s3 = tqdm(range(0, len(df), CONFIG['batch_size']), desc="Stage3 推論中")
        for i in pbar_s3:
            batch_df = df.iloc[i : i + CONFIG['batch_size']]
            s3_cond_values = torch.tensor(batch_df['weekday'].values, dtype=torch.float32).to(device)
            s3_cond_norm = (s3_cond_values - s3_cond_norm_stats['mean']) / s3_cond_norm_stats['std']
            s3_cond2_grid = s3_cond_norm.view(-1, 1, 1, 1, 1).expand(-1, 1, CONFIG['D'], CONFIG['H'], CONFIG['W'])

            s3_cond1_batch = torch.from_numpy(stage2_outputs_norm[i : i + len(batch_df)]).to(device)
            
            with torch.no_grad():
                s3_output_norm = stage3_model.sample(batch_size=len(batch_df), cond1_grid=s3_cond1_batch, cond2_grid=s3_cond2_grid)
            
            s3_output_denorm = s3_output_norm * s3_target_norm_stats['std'] + s3_target_norm_stats['mean']
            s3_output_denorm = torch.clamp(s3_output_denorm, min=0)
            all_s3_outputs.append(s3_output_denorm.cpu().numpy())
        
        simulated_flow_data_raw = np.concatenate(all_s3_outputs, axis=0)
        simulated_flow_data = simulated_flow_data_raw.squeeze(1).squeeze(1) # (N, H, W)
        np.save(s3_output_cache_path, simulated_flow_data)
        logger.info(f"Stage3 推論完成並已儲存快取: {s3_output_cache_path}")

    logger.info(f"模型推論完成。最終模擬資料形狀: {simulated_flow_data.shape}")
#%%
# --- 6. 計算累計正規化誤差 ---
logger.info("===== 步驟 6: 計算累計正規化誤差 =====")
normalized_error = (simulated_flow_data - actual_flow_data) / grid_stds
accumulated_error = np.sum(normalized_error, axis=0)
logger.info(f"累計誤差計算完成。形狀: {accumulated_error.shape}")
logger.info(f"累計誤差統計: Min={np.min(accumulated_error):.2f}, Max={np.max(accumulated_error):.2f}, Mean={np.mean(accumulated_error):.2f}")

# --- 7. 分析一: 累計正規化誤差 (Normalized Error) ---
logger.info("="*20 + " 分析一: 累計正規化誤差 " + "="*20)
# 7.1 計算
error_flat = accumulated_error.flatten()
grid_indices = np.arange(len(error_flat))
error_df = pd.DataFrame({'grid_idx': grid_indices, 'normalized_error_sum': error_flat})

# 7.2 視覺化 (連續圖)
max_abs_norm_error = np.max(np.abs(accumulated_error))
plot_grid_map(
    grid_values=accumulated_error,
    title="Accumulated Normalized Error Distribution",
    output_filename="accumulated_normalized_error_map.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap='coolwarm',
    cbar_label="Accumulated Normalized Error",
    vmin=-max_abs_norm_error, vmax=max_abs_norm_error
)
    
# 7.3 分組與視覺化 (分類圖)
top_index = 50 
pos_errors = error_df[error_df['normalized_error_sum'] >= 0].sort_values(by='normalized_error_sum', ascending=False)
neg_errors = error_df[error_df['normalized_error_sum'] < 0].sort_values(by='normalized_error_sum', ascending=True)
top_pos = pos_errors.head(top_index); other_pos = pos_errors.iloc[top_index:]
top_neg = neg_errors.head(top_index); other_neg = neg_errors.iloc[top_index:]
group_grid_norm = np.full_like(error_flat, -1, dtype=int)
group_grid_norm[top_pos.index] = 0; group_grid_norm[other_pos.index] = 1
group_grid_norm[top_neg.index] = 2; group_grid_norm[other_neg.index] = 3
error_df['group_norm_error'] = group_grid_norm
group_labels_norm = {
    0: f'Norm Error: Positive Top {top_index}', 1: 'Norm Error: Positive Others',
    2: f'Norm Error: Negative Top {top_index}', 3: 'Norm Error: Negative Others',
}
error_df['group_norm_error_label'] = error_df['group_norm_error'].map(group_labels_norm)
custom_colors = ['darkred', 'lightcoral', 'blue', 'lightblue']
plot_grid_map(
    grid_values=group_grid_norm.reshape(CONFIG['H'], CONFIG['W']),
    title="Accumulated Normalized Error Grouping Map",
    output_filename="accumulated_normalized_error_grouping_map.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap=custom_colors,
    is_categorical=True, cat_labels=[group_labels_norm[i] for i in sorted(group_labels_norm.keys())]
)


# --- 8. 分析二: 標準差誤差時數 (StdDev Exceedance Hours) ---
logger.info("="*20 + " 分析二: 標準差誤差時數 " + "="*20)
# 8.1 計算
pos_exceed_counts = np.sum(normalized_error > 1.0, axis=0).flatten()
neg_exceed_counts = np.sum(normalized_error < -1.0, axis=0).flatten()
exceed_df = pd.DataFrame({'grid_idx': grid_indices, 'pos_std_exceed_hours': pos_exceed_counts, 'neg_std_exceed_hours': neg_exceed_counts})
dominant_type_exceed = np.where(exceed_df['pos_std_exceed_hours'] >= exceed_df['neg_std_exceed_hours'], 'pos', 'neg')
signed_dominant_hours = np.where(dominant_type_exceed == 'pos', exceed_df['pos_std_exceed_hours'], -exceed_df['neg_std_exceed_hours'])
exceed_df['signed_dominant_std_hours'] = signed_dominant_hours
exceed_df['dominant_type_std_exceed'] = dominant_type_exceed

# 8.2 視覺化 (連續圖)
max_abs_exceed = np.max(np.abs(exceed_df['signed_dominant_std_hours'].values))
plot_grid_map(
    grid_values=exceed_df['signed_dominant_std_hours'].values.reshape(CONFIG['H'], CONFIG['W']),
    title="Distribution of Directional StdDev Exceedance Hours",
    output_filename="directional_std_exceedance_hours_map.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap='coolwarm',
    is_categorical=False, cbar_label="Directional Count of Hours Exceeding 1 StdDev",
    vmin=-max_abs_exceed, vmax=max_abs_exceed
)

# 8.3 分組與視覺化 (分類圖)
top_n_exceed = 50
exceed_df['dominant_std_hours_for_grouping'] = exceed_df[['pos_std_exceed_hours', 'neg_std_exceed_hours']].max(axis=1)
pos_dom_exceed = exceed_df[exceed_df['dominant_type_std_exceed'] == 'pos'].sort_values(by='dominant_std_hours_for_grouping', ascending=False)
neg_dom_exceed = exceed_df[exceed_df['dominant_type_std_exceed'] == 'neg'].sort_values(by='dominant_std_hours_for_grouping', ascending=False)
top_pos_exceed = pos_dom_exceed.head(top_n_exceed); other_pos_exceed = pos_dom_exceed.iloc[top_n_exceed:]
top_neg_exceed = neg_dom_exceed.head(top_n_exceed); other_neg_exceed = neg_dom_exceed.iloc[top_n_exceed:]
group_grid_exceed = np.full_like(error_flat, -1, dtype=int)
group_grid_exceed[top_pos_exceed.index] = 0; group_grid_exceed[other_pos_exceed.index] = 1
group_grid_exceed[top_neg_exceed.index] = 2; group_grid_exceed[other_neg_exceed.index] = 3
exceed_df['group_std_exceed_hours'] = group_grid_exceed
group_labels_exceed = {
    0: f'StdExceed: Positive Top {top_n_exceed}', 1: 'StdExceed: Positive Others',
    2: f'StdExceed: Negative Top {top_n_exceed}', 3: 'StdExceed: Negative Others',
}
exceed_df['group_std_exceed_hours_label'] = exceed_df['group_std_exceed_hours'].map(group_labels_exceed)
plot_grid_map(
    grid_values=group_grid_exceed.reshape(CONFIG['H'], CONFIG['W']),
    title="StdDev Exceedance Count Grouping Map",
    output_filename="std_dev_exceedance_map.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap=custom_colors,
    is_categorical=True, cat_labels=[group_labels_exceed[i] for i in sorted(group_labels_exceed.keys())]
)

# --- 9. 分析三: 累計原始誤差 (Raw Error) ---
logger.info("="*20 + " 分析三: 累計原始誤差 (已排除 1% 極端值) " + "="*20)
# 9.1 計算
raw_error_full = simulated_flow_data - actual_flow_data
# lower_bound = np.percentile(raw_error_full, 1); upper_bound = np.percentile(raw_error_full, 99)
# logger.info(f"原始誤差的 1% 分位數: {lower_bound:.2f}, 99% 分位數: {upper_bound:.2f}")
# trimmed_raw_error = np.where((raw_error_full > lower_bound) & (raw_error_full < upper_bound), raw_error_full, 0)
# logger.info("已將超出 1%-99% 範圍的原始誤差值設為 0。")
# pos_raw_error_sum = np.sum(np.maximum(0, trimmed_raw_error), axis=0).flatten()
# neg_raw_error_sum = np.sum(np.minimum(0, trimmed_raw_error), axis=0).flatten()
pos_raw_error_sum = np.sum(np.maximum(0, raw_error_full), axis=0).flatten()
neg_raw_error_sum = np.sum(np.minimum(0, raw_error_full), axis=0).flatten()
mae_df = pd.DataFrame({'grid_idx': grid_indices, 'pos_raw_error_sum': pos_raw_error_sum, 'neg_raw_error_sum': neg_raw_error_sum})
dominant_type_mae = np.where(mae_df['pos_raw_error_sum'] >= abs(mae_df['neg_raw_error_sum']), 'pos', 'neg')
mae_df['signed_raw_error_sum'] = mae_df['pos_raw_error_sum'] + mae_df['neg_raw_error_sum']
mae_df['dominant_type_mae'] = dominant_type_mae

# 9.2 削頂處理
if len(mae_df) > 1:
    sorted_mae = mae_df['signed_raw_error_sum'].sort_values(ascending=False)
    highest_val = sorted_mae.iloc[0]; second_highest_val = sorted_mae.iloc[1]
    if highest_val > second_highest_val:
        highest_idx = sorted_mae.index[0]
        mae_df.loc[highest_idx, 'signed_raw_error_sum'] = second_highest_val
        logger.info(f"分析三: 已將最高的累計原始誤差值 {highest_val:.2f} 修改為第二高的值 {second_highest_val:.2f}。")

# 9.3 視覺化 (連續圖)
raw_error_clipped = mae_df['signed_raw_error_sum'].values
max_abs_raw_error = np.max(np.abs(raw_error_clipped))
plot_grid_map(
    grid_values=raw_error_clipped.reshape(CONFIG['H'], CONFIG['W']),
    title="Accumulated Raw Error Distribution",
    output_filename="accumulated_raw_error_map_trimmed.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap='coolwarm',
    is_categorical=False, cbar_label="Accumulated Raw Error (Prediction - Actual)",
    vmin=-max_abs_raw_error, vmax=max_abs_raw_error
)

# 9.4 分組與視覺化 (分類圖)
top_n_mae = 50
pos_dom_mae = mae_df[mae_df['dominant_type_mae'] == 'pos'].sort_values(by='signed_raw_error_sum', ascending=False)
neg_dom_mae = mae_df[mae_df['dominant_type_mae'] == 'neg'].sort_values(by='signed_raw_error_sum', ascending=True)
top_pos_mae = pos_dom_mae.head(top_n_mae); other_pos_mae = pos_dom_mae.iloc[top_n_mae:]
top_neg_mae = neg_dom_mae.head(top_n_mae); other_neg_mae = neg_dom_mae.iloc[top_n_mae:]
group_grid_mae = np.full_like(error_flat, -1, dtype=int)
group_grid_mae[top_pos_mae.index] = 0; group_grid_mae[other_pos_mae.index] = 1
group_grid_mae[top_neg_mae.index] = 2; group_grid_mae[other_neg_mae.index] = 3
mae_df['group_raw_error'] = group_grid_mae
group_labels_mae = {
    0: f'Raw Error: Positive Top {top_n_mae}', 1: 'Raw Error: Positive Others',
    2: f'Raw Error: Negative Top {top_n_mae}', 3: 'Raw Error: Negative Others',
}
mae_df['group_raw_error_label'] = mae_df['group_raw_error'].map(group_labels_mae)
plot_grid_map(
    grid_values=group_grid_mae.reshape(CONFIG['H'], CONFIG['W']),
    title="Raw Error Grouping Map (1% Trimmed)",
    output_filename="raw_error_grouping_map_trimmed.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap=custom_colors,
    is_categorical=True, cat_labels=[group_labels_mae[i] for i in sorted(group_labels_mae.keys())]
)

# --- 10. 分析四: 原始誤差時數 (Raw Exceedance Hours) ---
logger.info("="*20 + " 分析四: 原始誤差超標時數 (閾值=100) " + "="*20)

# 10.1 計算
logger.info("===== 步驟 10.1: 計算原始誤差超過 100 的時數 =====")
raw_exceed_threshold = 100
pos_raw_exceed_counts = np.sum(raw_error_full > raw_exceed_threshold, axis=0).flatten()
neg_raw_exceed_counts = np.sum(raw_error_full < -raw_exceed_threshold, axis=0).flatten()
raw_exceed_df = pd.DataFrame({
    'grid_idx': grid_indices,
    'pos_raw_exceed_hours': pos_raw_exceed_counts,
    'neg_raw_exceed_hours': neg_raw_exceed_counts,
})
dominant_type_raw_exceed = np.where(raw_exceed_df['pos_raw_exceed_hours'] >= raw_exceed_df['neg_raw_exceed_hours'], 'pos', 'neg')
signed_dominant_raw_exceed = np.where(dominant_type_raw_exceed == 'pos', raw_exceed_df['pos_raw_exceed_hours'], -raw_exceed_df['neg_raw_exceed_hours'])
raw_exceed_df['signed_dominant_raw_exceed'] = signed_dominant_raw_exceed
raw_exceed_df['dominant_type_raw_exceed'] = dominant_type_raw_exceed

# 10.2 視覺化 (連續圖)
logger.info("===== 步驟 10.2: 視覺化帶方向性的原始誤差超標時數分佈圖 =====")
max_abs_raw_exceed = np.max(np.abs(raw_exceed_df['signed_dominant_raw_exceed'].values))
plot_grid_map(
    grid_values=raw_exceed_df['signed_dominant_raw_exceed'].values.reshape(CONFIG['H'], CONFIG['W']),
    title=f"Distribution of Directional Raw Error Exceedance Hours (Threshold={raw_exceed_threshold})",
    output_filename="directional_raw_exceedance_hours_map.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap='coolwarm',
    is_categorical=False, cbar_label=f"Directional Count of Hours |Error| > {raw_exceed_threshold}",
    vmin=-max_abs_raw_exceed, vmax=max_abs_raw_exceed
)

# 10.3 分組與視覺化 (分類圖)
logger.info("===== 步驟 10.3: 依原始誤差超標時數分組並視覺化 =====")
top_n_raw_exceed = 50
raw_exceed_df['dominant_hours_for_grouping'] = raw_exceed_df[['pos_raw_exceed_hours', 'neg_raw_exceed_hours']].max(axis=1)
pos_dom_raw_exceed = raw_exceed_df[raw_exceed_df['dominant_type_raw_exceed'] == 'pos'].sort_values(by='dominant_hours_for_grouping', ascending=False)
neg_dom_raw_exceed = raw_exceed_df[raw_exceed_df['dominant_type_raw_exceed'] == 'neg'].sort_values(by='dominant_hours_for_grouping', ascending=False)
top_pos_raw_exceed = pos_dom_raw_exceed.head(top_n_raw_exceed); other_pos_raw_exceed = pos_dom_raw_exceed.iloc[top_n_raw_exceed:]
top_neg_raw_exceed = neg_dom_raw_exceed.head(top_n_raw_exceed); other_neg_raw_exceed = neg_dom_raw_exceed.iloc[top_n_raw_exceed:]

group_grid_raw_exceed = np.full_like(error_flat, -1, dtype=int)
group_grid_raw_exceed[top_pos_raw_exceed.index] = 0; group_grid_raw_exceed[other_pos_raw_exceed.index] = 1
group_grid_raw_exceed[top_neg_raw_exceed.index] = 2; group_grid_raw_exceed[other_neg_raw_exceed.index] = 3
raw_exceed_df['group_raw_exceed_hours'] = group_grid_raw_exceed

group_labels_raw_exceed = {
    0: f'RawExceed: Positive Top {top_n_raw_exceed}', 1: 'RawExceed: Positive Others',
    2: f'RawExceed: Negative Top {top_n_raw_exceed}', 3: 'RawExceed: Negative Others',
}
raw_exceed_df['group_raw_exceed_hours_label'] = raw_exceed_df['group_raw_exceed_hours'].map(group_labels_raw_exceed)

plot_grid_map(
    grid_values=group_grid_raw_exceed.reshape(CONFIG['H'], CONFIG['W']),
    title=f"Raw Error Exceedance Count Grouping Map (Threshold={raw_exceed_threshold})",
    output_filename="raw_exceedance_grouping_map.png",
    config=CONFIG, grid_map_info=grid_map_info, cmap=custom_colors,
    is_categorical=True, cat_labels=[group_labels_raw_exceed[i] for i in sorted(group_labels_raw_exceed.keys())]
)

# --- 11. 匯出合併後的 Excel ---
logger.info("===== 步驟 11: 匯出所有分析結果至 Excel =====")
# 合併四個分析的 DataFrame
df_list = [error_df, exceed_df, mae_df, raw_exceed_df]
final_df = df_list[0]
for df_to_merge in df_list[1:]:
    # 避免重複的 'grid_idx' 欄位
    final_df = pd.merge(final_df, df_to_merge.drop(columns=[col for col in df_to_merge.columns if col in final_df.columns and col != 'grid_idx']), on='grid_idx', how='left')

# 附加地理資訊
rc_map = grid_map_info["grid_idx_to_rc_map"]
if rc_map:
    final_df['R'] = final_df['grid_idx'].apply(lambda idx: rc_map.get(idx, (-1, -1))[0])
    final_df['C'] = final_df['grid_idx'].apply(lambda idx: rc_map.get(idx, (-1, -1))[1])
else:
    final_df['R'] = -1; final_df['C'] = -1

sensor_coords_df = pd.DataFrame(grid_map_info["selected_sensor_info"])[['name', 'lon', 'lat']]
flow_cols_df = pd.DataFrame({'name': flow_columns}).reset_index().rename(columns={'index': 'grid_idx'})
excel_df = pd.merge(final_df, flow_cols_df, on='grid_idx', how='left')
excel_df = pd.merge(excel_df, sensor_coords_df, on='name', how='left')

# 整理最終輸出的欄位
output_columns = {
    'grid_idx': '網格ID', 'R': 'R', 'C': 'C', 'lon': '經度', 'lat': '緯度',
    # 分析一
    'normalized_error_sum': '累計正規化誤差',
    'group_norm_error_label': '分組(正規化誤差)',
    # 分析二
    'signed_dominant_std_hours': '方向性主要超標時數(Std)',
    'group_std_exceed_hours_label': '分組(超標時數Std)',
    # 分析三
    'signed_raw_error_sum': '累計原始誤差(1%修剪&削頂)',
    'group_raw_error_label': '分組(原始誤差)',
    # 分析四
    'signed_dominant_raw_exceed': '方向性主要超標時數(Raw)',
    'group_raw_exceed_hours_label': '分組(超標時數Raw)'
}
excel_output = excel_df[[col for col in output_columns.keys() if col in excel_df.columns]].rename(columns=output_columns)

excel_path = os.path.join(CONFIG['output_dir'], 'stage3_error_analysis_4_ways.xlsx')
excel_output.to_excel(excel_path, index=False)
logger.info(f"四種分析方法的合併結果已成功匯出至: {excel_path}")

logger.info("===== 全部分析流程結束 =====")
# %%

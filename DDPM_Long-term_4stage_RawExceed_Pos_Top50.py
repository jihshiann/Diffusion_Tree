#%%
import os
import re
import math
import json
import logging
import random
import numpy as np
import pandas as pd
import scipy.linalg # 用於 FID
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.optimize import linear_sum_assignment # 用於匈牙利演算法
from typing import Optional, Tuple, List, Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from enum import Enum

# ==============================================================================
# 組態設定
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    # --- 資料參數 ---
    "data_path": r"C:\thesis\code\Taipei_CF\all_merged.csv", # 資料路徑
    "H": 20, # 網格高度
    "W": 20, # 網格寬度
    "D": 1,  # 網格深度 (流量圖為1)

    # --- 模型架構參數 (basemodel, stage2_model, stage3_model 共用) ---
    "image_channels": 1,      # 主要資料(流量圖)的通道數
    "base_channels_unet": 16,   # UNet3D 的基礎通道數
    "unet_dropout_rate": 0.1,
    "time_emb_dim": 64,        # 時間嵌入維度
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 / UNet中與x_t合併的維度

    # === Basemodel 相關 (用於載入並決定其原始條件處理方式) ===
    # Basemodel 的 condition_processor 輸入通道數 (通常是2，因為它內部將小時、假日轉為2個網格)
    "basemodel_checkpoint": 
    r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_long-term\best_ddpm_model_during_training.pth", # Basemodel檢查點

    # === Stage2 特定配置 ===
    "stage2_new_condition_feature_column": "時", # Stage2 新條件的欄位名
    "stage2_new_conditional_operator": "<=",         # Stage2 新條件的運算符
    "stage2_new_conditional_value": 20,             # Stage2 新條件的閾值
    "stage2_model_name": "stage2_HourLe20",    # 第二階段模型的名稱
    "stage2_checkpoint_path": "best_stage2_model_hour_le_20.pth", # Stage2 模型的檢查點檔名 (相對路徑)

    # === Stage3 特定配置 ===
    "stage3_new_condition_feature_column": "weekday", # Stage3 新條件的欄位名 
    "stage3_new_conditional_operator": "<=",         # Stage3 新條件的運算符
    "stage3_new_conditional_value": 4,             # Stage3 新條件的閾值
    "stage3_model_name": "Stage3_WeekdayLe4",    # 第三階段模型的名稱
    "stage3_checkpoint_path": "best_stage3_model_Weekday_le_4.pth", # Stage3 模型的檢查點檔名 (相對路徑)

    # === Stage4 特定配置 ===
    "stage4_config": {
        # --- 模式開關: 'event' 或 'feature' ---
        "mode": "feature", 

        # --- 活動模式參數 (mode='event' 時啟用) ---
        "event_params": {
            "model_name": "stage4_arenaEvents",
            "checkpoint_path": "best_stage4_model_arena_events.pth",
            "event_filter": {
                "file_path": r"C:\thesis\code\Taipei_CF\ArenaEvents.xlsx",
                "year_col": "年",
                "month_col": "月",
                "day_col": "日"
            },
            # 這個欄位的值會被當作模型的條件輸入
            "grid_feature_source_column": "date_combined" 
        },

        # --- 特徵模式參數 (mode='feature' 時啟用) ---
        "feature_params": {
            "model_name": "Stage4_TotalCloudCoverM0",
            "checkpoint_path": "best_stage4_model_total_cloud_m_0.pth",
            # 這三個欄位用來篩選資料
            "new_condition_feature_column": "總雲量", 
            "new_conditional_operator": ">",
            "new_conditional_value": 0,
            # 這個欄位的值會被當作模型的條件輸入
            "grid_feature_source_column": "總雲量" 
        }
    },
    "baseline_model_path" : r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_baseline\Baseline_TotalCloudCoverM0\best_baseline_model_total_cloud_m_0.pth",
    "baseline_feature_columns": [
        "時", 
        "holiday", 
        "weekday", 
        "總雲量", 
    ],

    "coordinate_filter": {
        "enabled": True, # 設為 True 來啟用此功能
        "file_path": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage3\Stage3_WeekdayLe4\analysis_error\raw_exceed_hours_group0_RawExceed_Pos_Top50.xlsx", # 包含 R 和 C 欄位的 Excel 檔案
        "r_col": "R", # Excel 中代表「列」的欄位名
        "c_col": "C"  # Excel 中代表「行」的欄位名
    },

    # --- DDPM 擴散參數 ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 訓練參數 ---
    "epochs": 128,
    "batch_size": 256,
    "lr": 1e-3,

    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "weight_decay": 1e-5,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 4,
    "lr_scheduler_min_lr": 1e-7,
    "early_stopping_patience": 8,

    # --- 評估參數 ---
    "eval_batch_size": 256,
    "fid_batch_size": 256,
    "fid_num_samples": 128,

    # --- 路徑與儲存 ---
    "save_dir_stage2": "results_ddpm_stage2",
    "save_dir_stage3": "results_ddpm_stage3",
    "save_dir": "results_ddpm_stage4", # 主結果儲存目錄的基礎名稱
    
    "train_split_ratio": 0.7,
    "val_split_ratio": 0.15,

    # --- 快取設定 ---
    "cache_dir_name": "model_outputs_cache", # 相對於 save_dir 的快取目錄名稱
    "cached_basemodel_outputs_for_s2_filename": "basemodel_outputs_for_s2_normalized.npy",
    "mape_threshold": 1.0
}

# 動態設定 Stage2 和 Stage3 的快取檔案名稱
CONFIG["cached_stage2_outputs_for_s3_filename"] = f"stage2_outputs_{CONFIG['stage2_model_name']}_for_s3_normalized.npy"
CONFIG["cached_stage3_outputs_for_s4_filename"] = f"stage3_outputs_{CONFIG['stage3_model_name']}_for_s4_normalized.npy"


# 根據當前活躍的最高階段來設定通用的 condition_input_channels
# 這主要影響模型實例化時 DDPM3D 的 condition_processor。
# 在訓練/採樣時，我們會明確傳遞該階段所需的條件網格數量。
# 這裡假設每個階段的 DDPM condition_processor 都期望2個輸入通道。
CONFIG["condition_input_channels"] = 2


# 更新/生成 Stage2 相關路徑
CONFIG["stage2_model_save_dir"] = os.path.join(CONFIG["save_dir_stage2"], CONFIG["stage2_model_name"])
os.makedirs(CONFIG["stage2_model_save_dir"], exist_ok=True)
CONFIG["stage2_checkpoint_full_path"] = os.path.join(CONFIG["stage2_model_save_dir"], CONFIG["stage2_checkpoint_path"])

# 更新/生成 Stage3 相關路徑
CONFIG["stage3_model_save_dir"] = os.path.join(CONFIG["save_dir_stage3"], CONFIG["stage3_model_name"])
os.makedirs(CONFIG["stage3_model_save_dir"], exist_ok=True)
CONFIG["stage3_checkpoint_full_path"] = os.path.join(CONFIG["stage3_model_save_dir"], CONFIG["stage3_checkpoint_path"])

# 更新/生成 Stage4 相關路徑
s4_mode_setup = CONFIG["stage4_config"]["mode"]
s4_params_setup = CONFIG["stage4_config"][f"{s4_mode_setup}_params"]

CONFIG["stage4_model_save_dir"] = os.path.join(CONFIG["save_dir"], s4_params_setup["model_name"])
os.makedirs(CONFIG["stage4_model_save_dir"], exist_ok=True)
CONFIG["stage4_checkpoint_full_path"] = os.path.join(CONFIG["stage4_model_save_dir"], s4_params_setup["checkpoint_path"])

# 建立快取目錄路徑
CONFIG["cache_dir_full_path"] = os.path.join(CONFIG["save_dir"], CONFIG["cache_dir_name"])
os.makedirs(CONFIG["cache_dir_full_path"], exist_ok=True)
logger.info(f"模型輸出快取將儲存於: {CONFIG['cache_dir_full_path']}")


CONFIG["cached_basemodel_mean"] = 0.0
CONFIG["cached_basemodel_std"] = 1.0
CONFIG["cached_basemodel_sorted_flow_columns"] = []

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])
logger.info(f"使用裝置: {CONFIG['device']}")
logger.info(f"Stage4 結果將儲存於: {CONFIG['stage4_model_save_dir']}")

if not os.path.exists(CONFIG["basemodel_checkpoint"]):
    logger.error(f"【【【警告】】】 Basemodel 檢查點路徑未設定或檔案不存在: {CONFIG['basemodel_checkpoint']}")
if not os.path.exists(CONFIG["stage2_checkpoint_full_path"]): 
    logger.error(f"【【【警告】】】 Stage2 檢查點路徑 (用於載入給Stage3) 未設定或檔案不存在: {CONFIG['stage2_checkpoint_full_path']}")
if not os.path.exists(CONFIG["stage3_checkpoint_full_path"]): 
    logger.error(f"【【【警告】】】 Stage3 檢查點路徑 (用於載入給Stage4) 未設定或檔案不存在: {CONFIG['stage3_checkpoint_full_path']}")
class ConditionMode(Enum):
    BASEMODEL = 1
    STAGE2 = 2
    STAGE3 = 3
    STAGE4 = 4
    BASELINE_EVAL = 5

#%%
# ==============================================================================
# UNet3D, DDPM3D
# ==============================================================================

# UNet3D 建構模組及 UNet3D 類別的預留位置
class SinusoidalTimeEmbedding(nn.Module):
    """正弦時間嵌入"""
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device; half_dim = self.dim // 2; emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]; emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DoubleConv3D(nn.Module):
    """(卷積3D -> BN -> SiLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, kernel_size: int = 3, padding: int = 1):
        super().__init__(); mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm3d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm3d(out_channels), nn.SiLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.double_conv(x)

class Down3D(nn.Module):
    """下採樣模組 (MaxPool3D -> DoubleConv3D)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), # 深度維度不壓縮
            DoubleConv3D(in_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.maxpool_conv(x)

class Up3D(nn.Module):
    """上採樣模組"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__(); self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True) # 深度維度不放大
            self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2)) # 深度維度不放大
            self.conv = DoubleConv3D(in_channels, out_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor: # x1 是上採樣的張量, x2 是殘差連接的張量
        x1 = self.up(x1)
        # 輸入大小: C D H W
        diffY = x2.size()[3] - x1.size()[3] # H
        diffX = x2.size()[4] - x1.size()[4] # W
        # 深度維度 (dim 2) 不需要填充
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, # W
                        diffY // 2, diffY - diffY // 2, # H
                        0, 0])                          # D
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """輸出卷積層 (1x1x1 Conv3D)"""
    def __init__(self, in_channels: int, out_channels: int): super().__init__(); self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net 模型，帶有正確的時間嵌入投影"""
    def __init__(self, input_image_channels: int, base_channels: int = 64, time_emb_dim: int = 256,
                 condition_encode_dim: Optional[int] = None, bilinear_upsample: bool = True, dropout_rate: float = 0.05):
        super().__init__()
        self.input_image_channels = input_image_channels
        self.condition_encode_dim = condition_encode_dim or 0

        # 共享的時間嵌入 MLP (輸出維度是 time_emb_dim)
        self.shared_time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        actual_in_channels = self.input_image_channels + self.condition_encode_dim
        
        # --- U-Net 結構 ---
        self.inc = DoubleConv3D(actual_in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear_upsample else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor) # Bottleneck 層的前一層
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, bilinear_upsample)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, bilinear_upsample)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, bilinear_upsample)
        self.up4 = Up3D(base_channels * 2, base_channels, bilinear_upsample)
        self.outc = OutConv3D(base_channels, self.input_image_channels)

        # --- 為每個需要添加時間嵌入的層級定義線性投影層 ---
        self.time_proj_inc = nn.Linear(time_emb_dim, base_channels)
        self.time_proj_down1 = nn.Linear(time_emb_dim, base_channels * 2)
        self.time_proj_down2 = nn.Linear(time_emb_dim, base_channels * 4)
        self.time_proj_down3 = nn.Linear(time_emb_dim, base_channels * 8)
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, base_channels * 16 // factor) # 對應 down4 的輸出 (bottleneck)

        self.time_proj_up1 = nn.Linear(time_emb_dim, base_channels * 8 // factor)
        self.time_proj_up2 = nn.Linear(time_emb_dim, base_channels * 4 // factor)
        self.time_proj_up3 = nn.Linear(time_emb_dim, base_channels * 2 // factor)
        self.time_proj_up4 = nn.Linear(time_emb_dim, base_channels)

    def _add_time_embedding(self, x: torch.Tensor, t_emb_projected: torch.Tensor) -> torch.Tensor:
        # t_emb_projected 應該已經是 (N, C_feature_map) 的形狀
        t_emb_expanded = t_emb_projected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x + t_emb_expanded

    def forward(self, x_t: torch.Tensor, time_steps: torch.Tensor, processed_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 首先計算共享的時間嵌入 (N, time_emb_dim)
        shared_t_emb = self.shared_time_mlp(time_steps)

        if processed_condition is not None:
            if x_t.shape[2:] != processed_condition.shape[2:]: # 檢查 D, H, W 是否一致
                raise ValueError(f"x_t DHW {x_t.shape[2:]} != processed_condition DHW {processed_condition.shape[2:]}")
            x_input = torch.cat((x_t, processed_condition), dim=1) # 沿通道維度合併
        else:
            x_input = x_t

        x1 = self.inc(x_input)
        x1 = self._add_time_embedding(x1, self.time_proj_inc(shared_t_emb))

        x2 = self.down1(x1)
        x2 = self._add_time_embedding(x2, self.time_proj_down1(shared_t_emb))

        x3 = self.down2(x2)
        x3 = self._add_time_embedding(x3, self.time_proj_down2(shared_t_emb))

        x4 = self.down3(x3)
        x4 = self._add_time_embedding(x4, self.time_proj_down3(shared_t_emb))

        x5 = self.down4(x4) # Bottleneck 特徵
        x5 = self._add_time_embedding(x5, self.time_proj_bottleneck(shared_t_emb)) # 使用對應的投影
        x5 = self.dropout(x5)

        x = self.up1(x5, x4) # x4 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up1(shared_t_emb))

        x = self.up2(x, x3) # x3 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up2(shared_t_emb))

        x = self.up3(x, x2) # x2 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up3(shared_t_emb))

        x = self.up4(x, x1) # x1 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up4(shared_t_emb))
        
        return self.outc(x)
def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """線性 beta 排程"""
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM3D(nn.Module):
    def __init__(self,
                 unet_model: UNet3D,
                 timesteps: int,
                 image_size: Tuple[int, int, int], # (D, H, W)
                 image_channels: int,
                 condition_input_channels: int,
                 condition_encode_dim: int,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device: str = "cuda"):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps
        self.image_size_D, self.image_size_H, self.image_size_W = image_size
        self.image_channels = image_channels
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.condition_processor = nn.Sequential(
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()
        ).to(device)
        self.logger.info(f"DDPM3D instance created. Condition processor expects {condition_input_channels} input channels.")

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        sact = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        soma_ct = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sact * x_start + soma_ct * noise

    def _prepare_original_conditional_input_grids(self,
                                            hour_scalars_batch: torch.Tensor,
                                            is_holiday_scalars_batch: torch.Tensor,
                                            ) -> torch.Tensor: # 輸出 (N, 2, D, H, W)
        batch_size = hour_scalars_batch.shape[0]
        if hour_scalars_batch.shape[0] != is_holiday_scalars_batch.shape[0]:
            self.logger.error(f"Batch size mismatch in _prepare_original_conditional_input_grids: hour_batch={hour_scalars_batch.shape[0]}, holiday_batch={is_holiday_scalars_batch.shape[0]}")
            raise ValueError("Batch sizes for hour and holiday scalars must match.")

        norm_hours = hour_scalars_batch.float().to(self.device) / 23.0
        holiday_values = is_holiday_scalars_batch.float().to(self.device)

        hour_grid_vals = norm_hours.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)
        holiday_grid_vals = holiday_values.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)

        hour_grids_t = hour_grid_vals.unsqueeze(1).unsqueeze(2)
        holiday_grids_t = holiday_grid_vals.unsqueeze(1).unsqueeze(2)

        if self.image_size_D != 1:
            hour_grids_t = hour_grids_t.repeat(1,1,self.image_size_D,1,1)
            holiday_grids_t = holiday_grids_t.repeat(1,1,self.image_size_D,1,1)

        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_original_conditional_input_grids: Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels.")

        final_stacked_grids = torch.cat((hour_grids_t, holiday_grids_t), dim=1)
        return final_stacked_grids.to(self.device)

    def _prepare_stage_condition_grids(self,
                                     condition_grid_1_batch: torch.Tensor,
                                     condition_grid_2_batch: torch.Tensor
                                     ) -> torch.Tensor:
        expected_single_grid_shape = (1, self.image_size_D, self.image_size_H, self.image_size_W)
        # 檢查第一個條件網格的通道數是否為1 (因為通常是單一來源的網格，如BM輸出)
        if condition_grid_1_batch.shape[1] != 1:
            self.logger.warning(f"Stage condition_grid_1_batch has {condition_grid_1_batch.shape[1]} channels, expected 1. Using as is.")

        # 檢查第二個條件網格的通道數是否為1 (因為通常是單一來源的網格，如新特徵網格)
        if condition_grid_2_batch.shape[1] != 1:
            self.logger.warning(f"Stage condition_grid_2_batch has {condition_grid_2_batch.shape[1]} channels, expected 1. Using as is.")

        # 確保空間維度 (D, H, W) 匹配
        if condition_grid_1_batch.shape[2:] != expected_single_grid_shape[1:] or \
           condition_grid_2_batch.shape[2:] != expected_single_grid_shape[1:]:
            self.logger.error(f"Stage condition input grid spatial dimensions (D,H,W) are incorrect or mismatched. "
                              f"Grid1 spatial: {condition_grid_1_batch.shape[2:]}, Grid2 spatial: {condition_grid_2_batch.shape[2:]}. "
                              f"Expected spatial: {expected_single_grid_shape[1:]}")

        
        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_stage_condition_grids: Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels by concatenating two 1-channel grids.")
        return torch.cat((condition_grid_1_batch, condition_grid_2_batch), dim=1)

    def p_losses(self, x_start_target_flow: torch.Tensor, t: torch.Tensor,
                 mode: ConditionMode, # 明確的模式參數
                 condition_args: Dict[str, Optional[torch.Tensor]], # 一個包含所有可能條件的字典
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:

        if noise is None: noise = torch.randn_like(x_start_target_flow)
        x_t_noisy_target = self.q_sample(x_start=x_start_target_flow, t=t, noise=noise)
        stacked_cond_grids: Optional[torch.Tensor] = None
        
        self.logger.debug(f"p_losses called with mode: {mode}, condition_args keys: {list(condition_args.keys())}")

        if mode == ConditionMode.BASEMODEL:
            hour_s = condition_args.get("hour_scalars_batch")
            is_hol_s = condition_args.get("is_holiday_scalars_batch")
            if hour_s is None or is_hol_s is None:
                raise ValueError("p_losses (Basemodel mode): Requires 'hour_scalars_batch' and 'is_holiday_scalars_batch' in condition_args.")
            # 檢查是否提供了其他不應存在的鍵 (可選，但更穩健)
            unexpected_keys = [k for k in condition_args if k not in ["hour_scalars_batch", "is_holiday_scalars_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Basemodel mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(hour_s, is_hol_s)
        
        elif mode == ConditionMode.STAGE2:
            bm_out = condition_args.get("basemodel_output_grid_batch")
            s2_new_feat = condition_args.get("stage2_new_condition_feature_grid_batch")
            if bm_out is None or s2_new_feat is None:
                raise ValueError("p_losses (Stage2 mode): Requires 'basemodel_output_grid_batch' and 'stage2_new_condition_feature_grid_batch' in condition_args.")
            unexpected_keys = [k for k in condition_args if k not in ["basemodel_output_grid_batch", "stage2_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Stage2 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(bm_out, s2_new_feat)

        elif mode == ConditionMode.STAGE3:
            s2_out = condition_args.get("stage2_output_grid_batch_for_s3")
            s3_new_feat = condition_args.get("stage3_new_condition_feature_grid_batch")
            if s2_out is None or s3_new_feat is None:
                raise ValueError("p_losses (Stage3 mode): Requires 'stage2_output_grid_batch_for_s3' and 'stage3_new_condition_feature_grid_batch' in condition_args.")
            unexpected_keys = [k for k in condition_args if k not in ["stage2_output_grid_batch_for_s3", "stage3_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Stage3 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s2_out, s3_new_feat)

        elif mode == ConditionMode.STAGE4: # 新增 Stage4 處理
            s3_out = condition_args.get("stage3_output_grid_batch_for_s4")
            s4_new_feat = condition_args.get("stage4_new_condition_feature_grid_batch")
            if s3_out is None or s4_new_feat is None:
                raise ValueError("p_losses (Stage4 mode): Requires 'stage3_output_grid_batch_for_s4' and 'stage4_new_condition_feature_grid_batch' in condition_args.")
            unexpected_keys = [k for k in condition_args if k not in ["stage3_output_grid_batch_for_s4", "stage4_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Stage4 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s3_out, s4_new_feat)
        else:
            raise ValueError(f"p_losses: Unsupported condition mode: {mode}")

        expected_cond_proc_input_channels = self.condition_processor[0].in_channels
        if stacked_cond_grids.shape[1] != expected_cond_proc_input_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for p_losses. "
                              f"ConditionProcessor expected {expected_cond_proc_input_channels} channels, "
                              f"but got {stacked_cond_grids.shape[1]}.")
        stacked_cond_grids = stacked_cond_grids.to(self.device)
        processed_condition = self.condition_processor(stacked_cond_grids)
        predicted_noise = self.model(x_t_noisy_target, t, processed_condition)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, batch_size: int,
               mode: ConditionMode, # 明確的模式參數
               condition_args: Dict[str, Optional[torch.Tensor]] # 一個包含所有可能條件的字典
               ) -> torch.Tensor:

        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)
        stacked_cond_grids: Optional[torch.Tensor] = None
        
        self.logger.debug(f"sample called with mode: {mode}, condition_args keys: {list(condition_args.keys())}")
        if mode == ConditionMode.BASELINE_EVAL:
                stacked_cond_grids = condition_args.get("direct_condition")
                if stacked_cond_grids is None:
                    raise ValueError("sample (BASELINE_EVAL mode): 需要在 condition_args 中提供 'direct_condition'。")
        elif mode == ConditionMode.BASEMODEL:
            hour_s = condition_args.get("hour_scalars_batch")
            is_hol_s = condition_args.get("is_holiday_scalars_batch")
            if hour_s is None or is_hol_s is None or hour_s.shape[0] != batch_size or is_hol_s.shape[0] != batch_size:
                raise ValueError("sample (Basemodel mode): Requires 'hour_scalars_batch' and 'is_holiday_scalars_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["hour_scalars_batch", "is_holiday_scalars_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Basemodel mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(hour_s, is_hol_s).to(self.device)
        
        elif mode == ConditionMode.STAGE2:
            bm_out = condition_args.get("basemodel_output_grid_batch")
            s2_new_feat = condition_args.get("stage2_new_condition_feature_grid_batch")
            if bm_out is None or s2_new_feat is None or bm_out.shape[0] != batch_size or s2_new_feat.shape[0] != batch_size:
                raise ValueError("sample (Stage2 mode): Requires 'basemodel_output_grid_batch' and 'stage2_new_condition_feature_grid_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["basemodel_output_grid_batch", "stage2_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Stage2 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(bm_out, s2_new_feat)

        elif mode == ConditionMode.STAGE3:
            s2_out = condition_args.get("stage2_output_grid_batch_for_s3")
            s3_new_feat = condition_args.get("stage3_new_condition_feature_grid_batch")
            if s2_out is None or s3_new_feat is None or s2_out.shape[0] != batch_size or s3_new_feat.shape[0] != batch_size:
                raise ValueError("sample (Stage3 mode): Requires 'stage2_output_grid_batch_for_s3' and 'stage3_new_condition_feature_grid_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["stage2_output_grid_batch_for_s3", "stage3_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Stage3 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s2_out, s3_new_feat)

        elif mode == ConditionMode.STAGE4: # 新增 Stage4 處理
            s3_out = condition_args.get("stage3_output_grid_batch_for_s4")
            s4_new_feat = condition_args.get("stage4_new_condition_feature_grid_batch")
            if s3_out is None or s4_new_feat is None or s3_out.shape[0] != batch_size or s4_new_feat.shape[0] != batch_size:
                raise ValueError("sample (Stage4 mode): Requires 'stage3_output_grid_batch_for_s4' and 'stage4_new_condition_feature_grid_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["stage3_output_grid_batch_for_s4", "stage4_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Stage4 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s3_out, s4_new_feat)
        else:
            raise ValueError(f"sample: Unsupported condition mode: {mode}")

        # 驗证準備好的條件網格形狀
        if stacked_cond_grids.shape[1] != self.condition_processor[0].in_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for sampling. "
                              f"ConditionProcessor expected {self.condition_processor[0].in_channels} channels, "
                              f"but got {stacked_cond_grids.shape[1]}.")
        processed_conditions = self.condition_processor(stacked_cond_grids)

        for i in reversed(range(0, self.timesteps)):
            t_tensor_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            betas_t = self._extract(self.betas, t_tensor_batch, img.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor_batch, img.shape)
            sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t_tensor_batch, img.shape)
            
            predicted_noise_from_model = self.model(img, t_tensor_batch, processed_conditions)
            
            model_mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise_from_model / sqrt_one_minus_alphas_cumprod_t)
            if i == 0:
                img = model_mean
            else:
                posterior_variance_t = self._extract(self.posterior_variance, t_tensor_batch, img.shape)
                noise_sample = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise_sample
        return img
    

def create_next_stage_model_from_previous_checkpoint(
    config_for_current_stage_and_global: Dict[str, Any], # 傳入包含所有配置的字典
    device: str,
    current_stage_mode: ConditionMode # 使用 Enum 作為參數
) -> 'DDPM3D': # 假設 DDPM3D 類別已在此文件或已導入
    
    current_stage_name_for_log = current_stage_mode.name # 例如 "STAGE2", "STAGE3"
    previous_stage_checkpoint_path = "" 
    previous_stage_name_for_log = ""

    # 根據 current_stage_mode 動態決定 previous_stage_checkpoint_path 和 previous_stage_name_for_log
    if current_stage_mode == ConditionMode.STAGE2:
        previous_stage_checkpoint_path = config_for_current_stage_and_global['basemodel_checkpoint']
        previous_stage_name_for_log = "Basemodel"
    elif current_stage_mode == ConditionMode.STAGE3:
        previous_stage_checkpoint_path = config_for_current_stage_and_global['stage2_checkpoint_full_path']
        previous_stage_name_for_log = ConditionMode.STAGE2.name # "STAGE2"
    elif current_stage_mode == ConditionMode.STAGE4:
        previous_stage_checkpoint_path = config_for_current_stage_and_global['stage3_checkpoint_full_path']
        previous_stage_name_for_log = ConditionMode.STAGE3.name # "STAGE3"
    # 根據需要為更多階段添加 elif
    else:
        # 通常不應該為 BASEMODEL 模式呼叫此函數，因為它沒有 "前一階段" 的檢查點來載入
        raise ValueError(f"不支援的 current_stage_mode '{current_stage_name_for_log}' 用於從前一階段創建模型，或缺少對應的檢查點路徑配置。")

    if not previous_stage_checkpoint_path or not os.path.exists(previous_stage_checkpoint_path):
        raise FileNotFoundError(f"為 {current_stage_name_for_log} 模式確定的前一階段檢查點路徑 '{previous_stage_checkpoint_path}' 無效或檔案不存在。")

    logger.info(f"從 {previous_stage_name_for_log} 檢查點 {previous_stage_checkpoint_path} 創建並初始化 {current_stage_name_for_log} 模型...")
    
    chkpt_previous = torch.load(previous_stage_checkpoint_path, map_location=device, weights_only=False)
    if 'ddpm_state_dict' not in chkpt_previous:
        raise KeyError(f"{previous_stage_name_for_log} 檢查點 {previous_stage_checkpoint_path} 中未找到 'ddpm_state_dict'。")
    
    previous_chkpt_config = chkpt_previous.get('config_snapshot_at_save', 
                                             chkpt_previous.get('config', config_for_current_stage_and_global))

    current_stage_unet = UNet3D(
        input_image_channels=previous_chkpt_config.get("image_channels", config_for_current_stage_and_global["image_channels"]),
        base_channels=previous_chkpt_config.get("base_channels_unet", config_for_current_stage_and_global["base_channels_unet"]),
        time_emb_dim=previous_chkpt_config.get("time_emb_dim", config_for_current_stage_and_global["time_emb_dim"]),
        condition_encode_dim=previous_chkpt_config.get("condition_encode_dim", config_for_current_stage_and_global["condition_encode_dim"]),
        dropout_rate=previous_chkpt_config.get("unet_dropout_rate", config_for_current_stage_and_global.get("unet_dropout_rate", 0.05))
    ).to(device)

    current_stage_config_key_prefix = current_stage_name_for_log.lower() 
    
    current_model_condition_input_channels = config_for_current_stage_and_global.get(
        f"{current_stage_config_key_prefix}_ddpm_condition_input_channels", 
        config_for_current_stage_and_global.get("condition_input_channels", 2) 
    )
    logger.info(f"{current_stage_name_for_log} 模型將使用 {current_model_condition_input_channels} 個條件輸入通道。")

    current_stage_model_instance = DDPM3D(
        unet_model=current_stage_unet,
        timesteps=config_for_current_stage_and_global.get("timesteps"),
        image_size=(config_for_current_stage_and_global.get("D"),
                      config_for_current_stage_and_global.get("H"),
                      config_for_current_stage_and_global.get("W")),
        image_channels=config_for_current_stage_and_global.get("image_channels"),
        condition_input_channels=current_model_condition_input_channels,
        condition_encode_dim=config_for_current_stage_and_global.get("condition_encode_dim"),
        beta_start=config_for_current_stage_and_global.get("beta_start"),
        beta_end=config_for_current_stage_and_global.get("beta_end"),
        device=device
    )

    logger.info(f"將 {previous_stage_name_for_log} 的權重載入到新的 {current_stage_name_for_log} 模型實例 (condition_input_channels={current_model_condition_input_channels})...")
    try:
        current_stage_model_instance.load_state_dict(chkpt_previous['ddpm_state_dict'])
        logger.info(f"{current_stage_name_for_log} 模型權重從 {previous_stage_name_for_log} 完整遷移完成。")
    except RuntimeError as e:
        logger.warning(f"直接載入 {previous_stage_name_for_log} state_dict 到 {current_stage_name_for_log} 模型失敗: {e}")
        logger.warning(f"這可能是因為 {current_stage_name_for_log} 模型的 condition_processor 與 {previous_stage_name_for_log} 的不同。")
        logger.warning("嘗試僅載入 UNet (model) 部分的權重，並重新初始化 condition_processor...")
        
        unet_state_dict_to_load = {k.replace('model.', ''): v 
                                   for k, v in chkpt_previous['ddpm_state_dict'].items() 
                                   if k.startswith('model.')}
        if not unet_state_dict_to_load:
            logger.error(f"無法從 {previous_stage_name_for_log} 的 state_dict 中提取 UNet (model.) 權重。遷移失敗。")
            # 這裡可以選擇拋出錯誤或返回未初始化權重的模型，取決於您的錯誤處理策略
            raise ValueError(f"無法從 {previous_stage_name_for_log} 的 state_dict 中提取 UNet 權重。")
            
        current_stage_model_instance.model.load_state_dict(unet_state_dict_to_load)
        logger.info(f"僅 UNet 權重從 {previous_stage_name_for_log} 遷移完成。")

        cs_cond_input_ch = current_model_condition_input_channels
        cs_cond_encode_dim = config_for_current_stage_and_global.get("condition_encode_dim")
        
        # 重新初始化 condition_processor
        current_stage_model_instance.condition_processor = nn.Sequential(
            nn.Conv3d(cs_cond_input_ch, cs_cond_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(cs_cond_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(cs_cond_encode_dim // 2, cs_cond_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(cs_cond_encode_dim), nn.SiLU()
        ).to(device)
        logger.info(f"{current_stage_name_for_log} 模型的 condition_processor 已使用 {cs_cond_input_ch} 輸入通道重新初始化。")

    return current_stage_model_instance
# --------------------------------------
# 數據處理相關
# --------------------------------------
def parse_lat_lon(column_name: str) -> tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

def create_condition_mask(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
    """
    根據指定的條件，為 DataFrame 創建一個布林遮罩。

    Args:
        df (pd.DataFrame): 要進行過濾的 DataFrame。
        column (str): 要應用條件的欄位名稱。
        operator (str): 比較運算符，例如 '<=', '>', '==', '!='。
        value (Any): 用於比較的閾值。

    Returns:
        pd.Series: 一個布林 Series，可用於過濾 DataFrame。
    """
    if column not in df.columns:
        raise ValueError(f"欄位 '{column}' 不存在於 DataFrame 中。")

    # 確保進行比較的欄位是數值類型，無法轉換的會變成 NaN
    series_vals = pd.to_numeric(df[column], errors='coerce')
    
    # 根據運算符創建遮罩
    if operator == "<=":
        mask = (series_vals <= float(value))
    elif operator == ">":
        mask = (series_vals > float(value))
    elif operator == "<":
        mask = (series_vals < float(value))
    elif operator == ">=":
        mask = (series_vals >= float(value))
    elif operator == "==":
        mask = (series_vals == float(value))
    elif operator == "!=":
        mask = (series_vals != float(value))
    else:
        raise ValueError(f"不支援的運算符: '{operator}'")

    # 處理因 to_numeric 轉換失敗而產生的 NaN 值，將它們視為不滿足條件
    mask = mask.fillna(False)
    return mask

class MultiStageDataset(Dataset):
    def __init__(self,
                 df_for_processing: pd.DataFrame,
                 config: Dict[str, Any],
                 original_sorted_flow_columns: List[str],
                 current_stage_mode: ConditionMode,
                 mode: str = 'train',
                 # --- Pre-computed model outputs (normalized) ---
                 basemodel_outputs_np: Optional[np.ndarray] = None,
                 s2_model_outputs_np: Optional[np.ndarray] = None,
                 s3_model_outputs_np: Optional[np.ndarray] = None,
                 # --- Norm stats for new features from PREVIOUS stage checkpoints/training ---
                 s2_new_cond_feature_norm_stats: Optional[Dict[str, float]] = None, #來自S2檢查點
                 s3_new_cond_feature_norm_stats: Optional[Dict[str, float]] = None, #來自S3檢查點 (若當前>S3) 或 S3訓練實例 (若當前=S3 val/test)
                 s4_new_cond_feature_norm_stats: Optional[Dict[str, float]] = None, #來自S4訓練實例 (若當前=S4 val/test)
                 # --- For CURRENT stage's val/test mode, from its training instance ---
                 current_stage_avg_flow_map_dict_from_train: Optional[Dict[Tuple, np.ndarray]] = None,
                 current_stage_target_norm_stats_from_train: Optional[Dict[str, float]] = None
                 # current_stage_new_cond_feature_norm_stats_from_train 已包含在 sX_new_cond_feature_norm_stats 中
                 ):
        super().__init__()
        self.df_processed = df_for_processing.reset_index(drop=True)
        self.config = config
        self.current_stage_mode_enum = current_stage_mode
        self.current_stage_name = self.current_stage_mode_enum.name
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.MultiStageDataset[{self.current_stage_name}][{self.mode}]")

        self.H = config["H"]
        self.W = config["W"]
        self.D = config.get("D", 1)
        self.image_channels_target = config.get("image_channels", 1)
        self.sorted_flow_columns = original_sorted_flow_columns

        # Store pre-computed outputs from previous stages
        self.basemodel_outputs_np = basemodel_outputs_np
        self.s2_model_outputs_np = s2_model_outputs_np
        self.s3_model_outputs_np = s3_model_outputs_np
        
        # Initialize all potential new condition attributes to None or empty
        for i in range(2, 5): # Assuming up to Stage4 for now
            setattr(self, f's{i}_new_cond_col_name', None)
            setattr(self, f's{i}_new_cond_original_values_np', None)
            setattr(self, f's{i}_new_cond_category_for_target_np', None)
            setattr(self, f'norm_stats_s{i}_new_cond_feature', None)

        self._process_basemodel_conditions()
        self._process_all_stage_new_conditions(
            s2_new_cond_feature_norm_stats,
            s3_new_cond_feature_norm_stats,
            s4_new_cond_feature_norm_stats
        )
        self._calculate_or_load_current_stage_targets(
            current_stage_avg_flow_map_dict_from_train,
            current_stage_target_norm_stats_from_train
        )

        self.logger.info(f"Dataset __init__ (mode={self.mode}, stage={self.current_stage_name}) COMPLETED.")

    def _get_original_cond_values(self, col_name: str) -> np.ndarray:
        if col_name not in self.df_processed.columns:
            self.logger.error(f"Dataset: 條件欄位 '{col_name}' (請求者: {self.current_stage_name}, mode: {self.mode}) 不在 DataFrame 的欄位 '{list(self.df_processed.columns)}' 中。")
            raise ValueError(f"Dataset: 條件欄位 '{col_name}' 不在 DataFrame 中。")
        return pd.to_numeric(self.df_processed[col_name], errors='coerce').values

    def _process_basemodel_conditions(self):
        if '時' not in self.df_processed.columns:
            raise KeyError(f"DataFrame 中找不到 '時' 欄位 (for {self.current_stage_name} BM conditions)。")
        self.hours_for_target_np = self.df_processed['時'].values.astype(int)
        if not ((self.hours_for_target_np >= 0) & (self.hours_for_target_np <= 23)).all():
            self.logger.warning(f"'時' 欄位包含不在 0-23 範圍內的值。請檢查數據。")
        self.hour_category_for_target_grouping_np = (self.hours_for_target_np > 8).astype(int)

        if 'holiday' not in self.df_processed.columns and 'hoilday' in self.df_processed.columns:
            self.df_processed.rename(columns={"hoilday": "holiday"}, inplace=True)
        if 'holiday' not in self.df_processed.columns:
            raise KeyError(f"DataFrame 中找不到 'holiday' 或 'hoilday' 欄位 (for {self.current_stage_name} BM conditions)。")
        self.is_holiday_for_target_np = self.df_processed['holiday'].astype(bool).astype(int).values
        self.logger.info(f"BM 條件 (小時, 假日) 處理完畢。")
        self.logger.info(f"  小時類別分佈 (0: <=8, 1: >8): {dict(zip(*np.unique(self.hour_category_for_target_grouping_np, return_counts=True)))}")
        self.logger.info(f"  假日類別分佈 (0: 非假日, 1: 假日): {dict(zip(*np.unique(self.is_holiday_for_target_np, return_counts=True)))}")


    def _process_all_stage_new_conditions(self,
                                         s2_stats_source, #來自S2 chkpt
                                         s3_stats_source, #來自S3 chkpt (若當前>S3) 或 S3訓練實例 (若當前=S3 val/test)
                                         s4_stats_source  #來自S4訓練實例 (若當前=S4 val/test)
                                         ):
        # Stage 2 New Condition (always processed as it's base for S3/S4 targets)
        self.s2_new_cond_col_name = self.config["stage2_new_condition_feature_column"]
        self.s2_new_cond_original_values_np = self._get_original_cond_values(self.s2_new_cond_col_name)
        self.s2_new_cond_category_for_target_np = self._calculate_category_vector(
            self.s2_new_cond_original_values_np, self.config["stage2_new_conditional_operator"],
            self.config["stage2_new_conditional_value"], self.s2_new_cond_col_name, "S2Cond"
        )
        if s2_stats_source is None:
            raise ValueError("必須提供 Stage2 新條件的正規化統計量 (s2_stats_source)。")
        self.norm_stats_s2_new_cond_feature = s2_stats_source
        self._log_normalized_scalar_stats(self.s2_new_cond_original_values_np, self.norm_stats_s2_new_cond_feature,
                                          self.s2_new_cond_col_name, f"{self.current_stage_name}: Stage2 新條件")

        # Stage 3 New Condition
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE3.value:
            self.s3_new_cond_col_name = self.config["stage3_new_condition_feature_column"]
            self.s3_new_cond_original_values_np = self._get_original_cond_values(self.s3_new_cond_col_name)
            self.s3_new_cond_category_for_target_np = self._calculate_category_vector(
                self.s3_new_cond_original_values_np, self.config["stage3_new_conditional_operator"],
                self.config["stage3_new_conditional_value"], self.s3_new_cond_col_name, "S3Cond"
            )
            if self.current_stage_mode_enum == ConditionMode.STAGE3 and self.mode == 'train':
                self.norm_stats_s3_new_cond_feature = self._calculate_norm_stats(
                    self.s3_new_cond_original_values_np, self.s3_new_cond_col_name, "S3 new cond (Train)"
                )
            else: # S3 val/test, or CurrentStage is S4 (needs S3 stats from S3 checkpoint)
                if s3_stats_source is None:
                     raise ValueError(f"模式 {self.current_stage_name}/{self.mode}: 必須提供 Stage3 新條件的正規化統計量。")
                self.norm_stats_s3_new_cond_feature = s3_stats_source
            self._log_normalized_scalar_stats(self.s3_new_cond_original_values_np, self.norm_stats_s3_new_cond_feature,
                                              self.s3_new_cond_col_name, f"{self.current_stage_name}: Stage3 新條件")

        # Stage 4 New Condition
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE4.value:
            if self.current_stage_mode_enum == ConditionMode.STAGE4:
                
                # --- 這是關鍵的修改點 ---
                # 從我們新設計的 CONFIG 結構中，讀取代表性特徵的欄位名稱
                s4_mode = self.config["stage4_config"]["mode"]
                s4_params = self.config["stage4_config"][f"{s4_mode}_params"]
                self.s4_new_cond_col_name = s4_params["grid_feature_source_column"]
                
                # 後續的程式碼邏輯幾乎不變，因為它們是基於 self.s4_new_cond_col_name 這個變數運作的
                
                # 根據新的欄位名稱，獲取組合特徵的原始數值 (例如 407, 408...)
                self.s4_new_cond_original_values_np = self._get_original_cond_values(self.s4_new_cond_col_name)
                
                # 對於專家模型，所有數據都屬於同一個目標類別，直接設為 0 即可
                # 這也避免了去讀取已經不存在的舊 CONFIG 鍵
                self.s4_new_cond_category_for_target_np = np.zeros(len(self.df_processed), dtype=int)
                self.logger.info(f"專家模型模式：所有 {len(self.df_processed)} 筆 Stage4 數據的目標類別均設為 0。")

                if self.mode == 'train':
                    # 在訓練集上，為我們的組合特徵計算正規化統計量 (mean 和 std)
                    self.norm_stats_s4_new_cond_feature = self._calculate_norm_stats(
                        self.s4_new_cond_original_values_np, self.s4_new_cond_col_name, "S4 new cond (Train)"
                    )
                else: # S4 的驗證集或測試集
                    if s4_stats_source is None:
                        raise ValueError("Stage4 val/test mode 需要從訓練集傳入 Stage4 新條件的正規化統計量。")
                    # 使用從訓練集傳過來的統計量
                    self.norm_stats_s4_new_cond_feature = s4_stats_source
                    
                # 記錄日誌，顯示組合特徵在正規化後的統計分佈
                self._log_normalized_scalar_stats(self.s4_new_cond_original_values_np, self.norm_stats_s4_new_cond_feature,
                                                  self.s4_new_cond_col_name, f"{self.current_stage_name}: Stage4 新條件")

    def _calculate_or_load_current_stage_targets(self, avg_flow_map_from_train, target_norm_stats_from_train):
        if self.mode == 'train':
            self.average_flow_map_dict_current_stage = self._calculate_target_flows_for_current_stage()
            if not self.average_flow_map_dict_current_stage:
                self.logger.warning(f"_calculate_target_flows_for_current_stage() for {self.current_stage_name} 返回了一個空字典。")
                self.norm_stats_current_stage_target = {'mean': 0.0, 'std': 1.0}
            else:
                all_maps = np.array(list(self.average_flow_map_dict_current_stage.values()))
                if all_maps.size > 0:
                    self.norm_stats_current_stage_target = self._calculate_norm_stats(
                        all_maps.flatten(), f"{self.current_stage_name} Target", f"{self.current_stage_name} Target"
                    )
                else:
                    self.logger.warning(f"average_flow_map_dict_current_stage for {self.current_stage_name} 中的值為空數組。使用預設目標統計量。")
                    self.norm_stats_current_stage_target = {'mean': 0.0, 'std': 1.0}
            self.logger.info(f"計算得到 {self.current_stage_name} 目標流量的專用正規化統計量: mean={self.norm_stats_current_stage_target['mean']:.4f}, std={self.norm_stats_current_stage_target['std']:.4f}")
        
        elif self.mode == 'val' or self.mode == 'test':
            if avg_flow_map_from_train is None or target_norm_stats_from_train is None:
                raise ValueError(f"{self.current_stage_name} {self.mode} mode 需要從訓練集傳入 avg_flow_map 和 target_norm_stats。")
            self.average_flow_map_dict_current_stage = avg_flow_map_from_train
            self.norm_stats_current_stage_target = target_norm_stats_from_train
            self.logger.info(f"已載入 {self.current_stage_name} 目標流量的預計算平均圖和專用正規化統計量: mean={self.norm_stats_current_stage_target['mean']:.4f}, std={self.norm_stats_current_stage_target['std']:.4f}")
        else:
            raise ValueError(f"未知的 Dataset mode: {self.mode}")

    def _log_normalized_scalar_stats(self, original_values_np: Optional[np.ndarray], 
                                     norm_stats: Optional[Dict[str, float]], 
                                     col_name: str, log_prefix: str):
        if original_values_np is None:
            self.logger.debug(f"{log_prefix} ('{col_name}'): 原始值數組為 None，跳過正規化統計。")
            return
        if norm_stats is None:
            self.logger.warning(f"{log_prefix} ('{col_name}'): 正規化統計量為 None，無法打印詳細統計。")
            return
            
        mean_stat = norm_stats.get('mean')
        std_stat = norm_stats.get('std')

        if mean_stat is not None and std_stat is not None:
            current_std_stat = std_stat if std_stat >= 1e-6 else 1.0
            
            normalized_values = np.array([
                (val - mean_stat) / current_std_stat if not np.isnan(val) else 0.0
                for val in original_values_np 
            ])
            if normalized_values.size > 0:
                self.logger.info(f"{log_prefix} ('{col_name}') 使用提供的統計量正規化後純量值統計: "
                                 f"MIN: {np.min(normalized_values):.4f}, MAX: {np.max(normalized_values):.4f}, "
                                 f"MEAN: {np.mean(normalized_values):.4f}, STD: {np.std(normalized_values):.4f}")
            else:
                self.logger.warning(f"{log_prefix} ('{col_name}') 數據為空，無法計算正規化後純量值統計。")
        else:
            self.logger.warning(f"{log_prefix} ('{col_name}') 的正規化統計量缺失 mean 或 std。")
            
    def _calculate_category_vector(self, values_np: np.ndarray, op: str, threshold: Any, col_name_for_log: str, cond_stage_log_prefix: str) -> np.ndarray:
        num_nan = np.isnan(values_np).sum()
        if num_nan > 0: self.logger.warning(f"{self.__class__.__name__}[{self.current_stage_name}] ({cond_stage_log_prefix}, mode={self.mode}): 欄位 '{col_name_for_log}' 包含 {num_nan} 個 NaN。比較時 NaN 通常結果為 False。")
        series_vals = pd.Series(values_np)
        try:
            thresh_val = float(threshold)
            cat_0_desc, cat_1_desc = "", ""
            if op == "<=": condition_met_mask = (series_vals <= thresh_val); cat_0_desc=f"'{col_name_for_log}' <= {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' > {thresh_val}"
            elif op == ">": condition_met_mask = (series_vals > thresh_val); cat_0_desc=f"'{col_name_for_log}' > {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' <= {thresh_val}"
            # ... (其他 op 判斷與原 Stage3Dataset 相同) ...
            elif op == "<": condition_met_mask = (series_vals < thresh_val); cat_0_desc=f"'{col_name_for_log}' < {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' >= {thresh_val}"
            elif op == ">=": condition_met_mask = (series_vals >= thresh_val); cat_0_desc=f"'{col_name_for_log}' >= {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' < {thresh_val}"
            elif op == "==": condition_met_mask = (series_vals == thresh_val); cat_0_desc=f"'{col_name_for_log}' == {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' != {thresh_val}"
            elif op == "!=": condition_met_mask = (series_vals != thresh_val); cat_0_desc=f"'{col_name_for_log}' != {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' == {thresh_val}"
            else:
                self.logger.warning(f"{self.__class__.__name__}[{self.current_stage_name}] ({cond_stage_log_prefix}, mode={self.mode}): 未明確處理運算符 '{op}' for column '{col_name_for_log}'，預設分類為 ({col_name_for_log} <= {thresh_val}) 為類別0。")
                condition_met_mask = (series_vals <= thresh_val); cat_0_desc=f"'{col_name_for_log}' <= {thresh_val} (預設)"; cat_1_desc=f"'{col_name_for_log}' > {thresh_val} (預設)"
            
            category_vector = (~condition_met_mask).astype(int) 
            self.logger.info(f"{self.__class__.__name__}[{self.current_stage_name}] ({cond_stage_log_prefix}, mode={self.mode}): 條件 ('{col_name_for_log}') 分類邏輯 -> 類別0 (主要條件滿足): {cat_0_desc}; 類別1 (不滿足): {cat_1_desc}")
            unique_cats, counts_cats = np.unique(category_vector, return_counts=True)
            self.logger.info(f"  - 分類 ('{col_name_for_log}') 分佈: {dict(zip(unique_cats, counts_cats))}")
            return category_vector
        except ValueError:
            self.logger.error(f"{self.__class__.__name__}[{self.current_stage_name}] ({cond_stage_log_prefix}, mode={self.mode}): 閾值 '{threshold}' for '{col_name_for_log}' 無法轉換為浮點數。所有樣本類別將設為0。")
            return np.zeros(len(self.df_processed), dtype=int)

    def _calculate_norm_stats(self, values_np: np.ndarray, col_name_for_log: str, data_source_description: str) -> Dict[str, float]:
        valid_values = values_np[~np.isnan(values_np)]
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
        else:
            self.logger.warning(f"{self.__class__.__name__}[{self.current_stage_name}] ({data_source_description}, {self.mode}): 欄位 '{col_name_for_log}' 中沒有有效的數值用於計算正規化統計量。將使用 mean=0, std=1。")
            return {'mean': 0.0, 'std': 1.0}
        if std_val < 1e-6:
            self.logger.warning(f"{self.__class__.__name__}[{self.current_stage_name}] ({data_source_description}, {self.mode}): 計算得到欄位 '{col_name_for_log}' 標準差 ({std_val:.4f}) 過小，將其設為 1.0。")
            std_val = 1.0
        return {'mean': mean_val, 'std': std_val}

    def __len__(self) -> int:
        return len(self.df_processed)

    def _calculate_target_flows_for_current_stage(self) -> Dict[Tuple, np.ndarray]:
        self.logger.info(f"為 {self.current_stage_name} 計算目標平均流量...")
        avg_flows: Dict[Tuple, np.ndarray] = {}
        flow_data = self.df_processed[self.sorted_flow_columns].values.astype(np.float32)
        
        grouping_data = {
            'hour_category': self.hour_category_for_target_grouping_np,
            'is_holiday': self.is_holiday_for_target_np
        }
        group_by_cols = ['hour_category', 'is_holiday']

        # 根據當前階段動態添加分組條件
        if hasattr(self, 's2_new_cond_category_for_target_np'):
            grouping_data['s2_cond_category'] = self.s2_new_cond_category_for_target_np
            if 's2_cond_category' not in group_by_cols : group_by_cols.append('s2_cond_category')
        
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE3.value and hasattr(self, 's3_new_cond_category_for_target_np'):
            grouping_data['s3_cond_category'] = self.s3_new_cond_category_for_target_np
            if 's3_cond_category' not in group_by_cols : group_by_cols.append('s3_cond_category')
        
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE4.value and hasattr(self, 's4_new_cond_category_for_target_np'):
            grouping_data['s4_cond_category'] = self.s4_new_cond_category_for_target_np
            if 's4_cond_category' not in group_by_cols : group_by_cols.append('s4_cond_category')
        
        grouping_df = pd.DataFrame(grouping_data)
        if grouping_df.empty:
            self.logger.warning(f"{self.current_stage_name} 目標: Grouping DataFrame 為空。")
            return {}
            
        actual_groupby_cols = [col for col in group_by_cols if col in grouping_df.columns]
        if not actual_groupby_cols:
            self.logger.error(f"{self.current_stage_name} 目標: 沒有有效的列用於 groupby。")
            return {}

        grouped = grouping_df.groupby(actual_groupby_cols, observed=False)
        if not grouped.groups or all(idx.empty for idx in grouped.groups.values()):
            self.logger.warning(f"{self.current_stage_name} 目標: 分組後 grouped.groups 為空或所有組都為空。")
            return {}

        self.logger.info(f"{self.current_stage_name} 目標: 各條件組合的資料筆數分佈如下 {tuple(actual_groupby_cols)}: count")
        for group_key, group_indices in grouped.indices.items():
            count = len(group_indices)
            if count == 0: continue
            # 確保 group_key 總是元組，即使只有一個元素，以保持字典鍵的一致性
            current_group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
            self.logger.info(f"  - 組合 {current_group_key_tuple}: {count} 筆資料")
            mean_flow_flat = np.nanmean(flow_data[group_indices], axis=0)
            mean_flow_flat[np.isnan(mean_flow_flat)] = 0
            avg_flows[current_group_key_tuple] = mean_flow_flat.reshape(self.H, self.W)
        
        self.logger.info(f"計算完成 {len(avg_flows)} 個 {self.current_stage_name} 條件的目標平均流量圖。")
        return avg_flows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # --- 1. 當前階段模型的目標 (正規化) ---
        grouping_values_for_key_list = [
            self.hour_category_for_target_grouping_np[idx],
            self.is_holiday_for_target_np[idx]
        ]
        if hasattr(self, 's2_new_cond_category_for_target_np'):
            grouping_values_for_key_list.append(self.s2_new_cond_category_for_target_np[idx])
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE3.value and hasattr(self, 's3_new_cond_category_for_target_np'):
            grouping_values_for_key_list.append(self.s3_new_cond_category_for_target_np[idx])
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE4.value and hasattr(self, 's4_new_cond_category_for_target_np'):
            grouping_values_for_key_list.append(self.s4_new_cond_category_for_target_np[idx])
        
        target_key_current_stage = tuple(grouping_values_for_key_list)
        
        target_avg_flow_np = self.average_flow_map_dict_current_stage.get(target_key_current_stage, 
                                                                          np.zeros((self.H, self.W), dtype=np.float32))
        target_mean_norm = self.norm_stats_current_stage_target['mean']
        target_std_norm = self.norm_stats_current_stage_target['std']
        if target_std_norm < 1e-6: target_std_norm = 1.0
        norm_target_current_stage_np = (target_avg_flow_np - target_mean_norm) / target_std_norm
        target_current_stage_tensor_norm = torch.from_numpy(norm_target_current_stage_np).float().reshape(
            self.image_channels_target, self.D, self.H, self.W
        )

        # --- 2. 當前階段模型的條件1 (前一階段輸出, 正規化) ---
        condition1_current_stage_tensor_norm: torch.Tensor
        if self.current_stage_mode_enum == ConditionMode.STAGE2:
            if self.basemodel_outputs_np is None: raise ValueError("Dataset for STAGE2 mode requires basemodel_outputs_np.")
            prev_out_sample = self.basemodel_outputs_np[idx]
        elif self.current_stage_mode_enum == ConditionMode.STAGE3:
            if self.s2_model_outputs_np is None: raise ValueError("Dataset for STAGE3 mode requires s2_model_outputs_np.")
            prev_out_sample = self.s2_model_outputs_np[idx]
        elif self.current_stage_mode_enum == ConditionMode.STAGE4:
            if self.s3_model_outputs_np is None: raise ValueError("Dataset for STAGE4 mode requires s3_model_outputs_np.")
            prev_out_sample = self.s3_model_outputs_np[idx]
        else:
            self.logger.error(f"無法為 {self.current_stage_name} 模式確定條件1的來源。")
            prev_out_sample = np.zeros((self.image_channels_target, self.D, self.H, self.W), dtype=np.float32)
        
        if prev_out_sample.shape[0] != self.image_channels_target : 
             prev_out_sample = prev_out_sample[0:self.image_channels_target, ...] 
        condition1_current_stage_tensor_norm = torch.from_numpy(prev_out_sample.astype(np.float32))

        # --- 3. 當前階段模型的條件2 (當前階段新特徵, 正規化) ---
        original_current_stage_cond_value: float
        current_stage_new_cond_norm_stats: Dict[str,float]

        if self.current_stage_mode_enum == ConditionMode.STAGE2:
            original_current_stage_cond_value = self.s2_new_cond_original_values_np[idx]
            current_stage_new_cond_norm_stats = self.norm_stats_s2_new_cond_feature
        elif self.current_stage_mode_enum == ConditionMode.STAGE3:
            original_current_stage_cond_value = self.s3_new_cond_original_values_np[idx]
            current_stage_new_cond_norm_stats = self.norm_stats_s3_new_cond_feature
        elif self.current_stage_mode_enum == ConditionMode.STAGE4:
            original_current_stage_cond_value = self.s4_new_cond_original_values_np[idx]
            current_stage_new_cond_norm_stats = self.norm_stats_s4_new_cond_feature
        else:
            self.logger.error(f"無法為 {self.current_stage_name} 模式確定條件2的來源。")
            original_current_stage_cond_value = 0.0 
            current_stage_new_cond_norm_stats = {'mean':0.0, 'std':1.0} 
            
        current_stage_cond_mean = current_stage_new_cond_norm_stats['mean']
        current_stage_cond_std = current_stage_new_cond_norm_stats['std']
        if current_stage_cond_std < 1e-6: current_stage_cond_std = 1.0
        normalized_current_stage_cond_value = (original_current_stage_cond_value - current_stage_cond_mean) / current_stage_cond_std \
            if not np.isnan(original_current_stage_cond_value) else 0.0
        condition2_current_stage_tensor_norm = torch.full(
            (1, self.D, self.H, self.W), float(normalized_current_stage_cond_value), dtype=torch.float32
        )

        # --- 4. 輔助數據，用於評估 Basemodel ---
        original_hour_scalar = torch.tensor(self.hours_for_target_np[idx], dtype=torch.long)
        original_is_holiday_scalar = torch.tensor(self.is_holiday_for_target_np[idx], dtype=torch.long)

        # --- 5. 輔助數據，用於評估 Stage2 模型 (BM輸出 + S2新特徵) ---
        bm_output_grid_for_s2eval = torch.empty(0) 
        if self.basemodel_outputs_np is not None:
            bm_out_s = self.basemodel_outputs_np[idx]
            if bm_out_s.shape[0] != 1: bm_out_s = bm_out_s[0:1, ...]
            bm_output_grid_for_s2eval = torch.from_numpy(bm_out_s.astype(np.float32))
        
        s2_new_feat_grid_for_s2eval = torch.empty(0)
        s2_original_feature_scalar = torch.tensor(0.0, dtype=torch.float32) 
        if hasattr(self, 's2_new_cond_original_values_np') and self.norm_stats_s2_new_cond_feature:
            orig_s2_val_eval = self.s2_new_cond_original_values_np[idx]
            s2_original_feature_scalar = torch.tensor(orig_s2_val_eval if not np.isnan(orig_s2_val_eval) else 0.0, dtype=torch.float32)
            s2_mean = self.norm_stats_s2_new_cond_feature['mean']
            s2_std = self.norm_stats_s2_new_cond_feature['std']
            if s2_std < 1e-6: s2_std = 1.0
            norm_s2_val_eval = (orig_s2_val_eval - s2_mean) / s2_std if not np.isnan(orig_s2_val_eval) else 0.0
            s2_new_feat_grid_for_s2eval = torch.full((1, self.D, self.H, self.W), float(norm_s2_val_eval), dtype=torch.float32)

        # --- 6. 輔助數據，用於評估 Stage3 模型 (S2輸出 + S3新特徵) ---
        s2_output_grid_for_s3eval = torch.empty(0)
        if self.s2_model_outputs_np is not None: # s2_model_outputs_np now general for "prev stage"
            s2_out_s = self.s2_model_outputs_np[idx]
            if s2_out_s.shape[0] != 1: s2_out_s = s2_out_s[0:1, ...]
            s2_output_grid_for_s3eval = torch.from_numpy(s2_out_s.astype(np.float32))

        s3_new_feat_grid_for_s3eval = torch.empty(0)
        s3_original_feature_scalar = torch.tensor(0.0, dtype=torch.float32)
        if hasattr(self, 's3_new_cond_original_values_np') and hasattr(self, 'norm_stats_s3_new_cond_feature') and self.norm_stats_s3_new_cond_feature:
            orig_s3_val_eval = self.s3_new_cond_original_values_np[idx]
            s3_original_feature_scalar = torch.tensor(orig_s3_val_eval if not np.isnan(orig_s3_val_eval) else 0.0, dtype=torch.float32)
            s3_mean = self.norm_stats_s3_new_cond_feature['mean']
            s3_std = self.norm_stats_s3_new_cond_feature['std']
            if s3_std < 1e-6: s3_std = 1.0
            norm_s3_val_eval = (orig_s3_val_eval - s3_mean) / s3_std if not np.isnan(orig_s3_val_eval) else 0.0
            s3_new_feat_grid_for_s3eval = torch.full((1, self.D, self.H, self.W), float(norm_s3_val_eval), dtype=torch.float32)

        # --- 7. 輔助數據，用於評估 Stage4 模型 (S3輸出 + S4新特徵) ---
        s3_output_grid_for_s4eval = torch.empty(0) # This is Cond1 for S4 model
        if self.current_stage_mode_enum.value >= ConditionMode.STAGE4.value: # Only if current stage is S4 or higher
            if self.s3_model_outputs_np is not None: # s3_model_outputs_np is prev_stage_output if current_stage=S4
                s3_out_s = self.s3_model_outputs_np[idx]
                if s3_out_s.shape[0] != 1: s3_out_s = s3_out_s[0:1, ...]
                s3_output_grid_for_s4eval = torch.from_numpy(s3_out_s.astype(np.float32))
            else: # This case should be handled if S4 needs S3 output
                 self.logger.warning(f"__getitem__ (idx {idx}): s3_model_outputs_np is None, S4 eval might lack S3 output.")


        s4_new_feat_grid_for_s4eval = torch.empty(0) # This is Cond2 for S4 model if S4 is being evaluated by a S5
        s4_original_feature_scalar = torch.tensor(0.0, dtype=torch.float32)
        if hasattr(self, 's4_new_cond_original_values_np') and hasattr(self, 'norm_stats_s4_new_cond_feature') and self.norm_stats_s4_new_cond_feature:
             orig_s4_val_eval = self.s4_new_cond_original_values_np[idx]
             s4_original_feature_scalar = torch.tensor(orig_s4_val_eval if not np.isnan(orig_s4_val_eval) else 0.0, dtype=torch.float32)
             s4_mean = self.norm_stats_s4_new_cond_feature['mean']
             s4_std = self.norm_stats_s4_new_cond_feature['std']
             if s4_std < 1e-6: s4_std = 1.0
             norm_s4_val_eval = (orig_s4_val_eval - s4_mean) / s4_std if not np.isnan(orig_s4_val_eval) else 0.0
             s4_new_feat_grid_for_s4eval = torch.full((1, self.D, self.H, self.W), float(norm_s4_val_eval), dtype=torch.float32)


        return (
            target_current_stage_tensor_norm,           # 0: 當前階段目標 (正規化)
            condition1_current_stage_tensor_norm,       # 1: 當前階段模型條件1 (前一階段輸出, 正規化)
            condition2_current_stage_tensor_norm,       # 2: 當前階段模型條件2 (當前階段新特徵, 正規化)
            
            original_hour_scalar,                       # 3: BM原始小時
            original_is_holiday_scalar,                 # 4: BM原始假日
            
            bm_output_grid_for_s2eval,                  # 5: BM輸出網格 (S2條件1, 用於評估S2)
            s2_new_feat_grid_for_s2eval,                # 6: S2新特徵網格 (S2條件2, 用於評估S2)
            
            s2_output_grid_for_s3eval,                  # 7: S2輸出網格 (S3條件1, 用於評估S3)
            s3_new_feat_grid_for_s3eval,                # 8: S3新特徵網格 (S3條件2, 用於評估S3)

            s3_output_grid_for_s4eval,                  # 9: S3輸出網格 (S4條件1, 用於評估S4)
            s4_new_feat_grid_for_s4eval,                # 10: S4新條件網格 (S4條件2, 用於評估S4)

            s2_original_feature_scalar,                 # 11: S2 原始新特徵純量
            s3_original_feature_scalar,                 # 12: S3 原始新特徵純量
            s4_original_feature_scalar                  # 13: S4 原始新特徵純量
        )

class BaselineDataset(Dataset):
    """為單階段 Baseline 模型設計的數據集類別。"""
    def __init__(self, df_for_processing, config, mode='train', 
                 norm_stats_from_train=None, target_info_from_train=None):
        super().__init__()
        self.df_processed = df_for_processing.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.BaselineDataset[{self.mode}]")
        self.H, self.W, self.D = config["H"], config["W"], config.get("D", 1)
        self.image_channels_target = config["image_channels"]
        self.sorted_flow_columns = config["cached_basemodel_sorted_flow_columns"]
        self._process_conditions(norm_stats_from_train)
        self._calculate_or_load_targets(target_info_from_train)
        self.logger.info(f"BaselineDataset (mode={self.mode}) 初始化完成，含 {len(self.df_processed)} 筆樣本。")

    def _get_original_cond_values(self, col_name):
        # 確保 'hoilday' 的拼寫錯誤被修正
        if col_name == 'holiday' and 'holiday' not in self.df_processed.columns and 'hoilday' in self.df_processed.columns:
            col_name = 'hoilday'
        return pd.to_numeric(self.df_processed[col_name], errors='coerce').values

    def _calculate_norm_stats(self, values_np, col_name):
        valid_values = values_np[~np.isnan(values_np)]
        mean, std = (np.mean(valid_values), np.std(valid_values)) if len(valid_values) > 0 else (0.0, 1.0)
        return {'mean': mean, 'std': std if std > 1e-6 else 1.0}

    def _process_conditions(self, norm_stats_from_train):
        self.feature_columns = self.config.get("baseline_feature_columns", [])
        self.original_values_dict = {col: self._get_original_cond_values(col) for col in self.feature_columns}
        if self.mode == 'train':
            raise NotImplementedError("BaselineDataset 在此腳本中僅用於測試模式")
        else:
            if norm_stats_from_train is None:
                raise ValueError("測試模式需要從 Baseline 檢查點傳入 cond_norm_stats。")
            self.norm_stats_dict = norm_stats_from_train

    def _calculate_or_load_targets(self, target_info_from_train):
        self.hour_category_for_target_grouping_np = (self.df_processed['時'].values > 8).astype(int)
        self.is_holiday_for_target_np = self.df_processed['holiday'].astype(bool).astype(int).values
        s2_vals = pd.to_numeric(self.df_processed[self.config["stage2_new_condition_feature_column"]], errors='coerce').values
        self.s2_cond_category_for_target_np = (~(pd.Series(s2_vals) <= self.config["stage2_new_conditional_value"])).astype(int)
        s3_vals = pd.to_numeric(self.df_processed[self.config["stage3_new_condition_feature_column"]], errors='coerce').values
        self.s3_cond_category_for_target_np = (~(pd.Series(s3_vals) <= self.config["stage3_new_conditional_value"])).astype(int)
        self.s4_cond_category_for_target_np = np.zeros(len(self.df_processed), dtype=int)
        
        if self.mode != 'test':
            raise NotImplementedError("BaselineDataset 在此腳本中僅用於測試模式")
        
        self.average_flow_map_dict = target_info_from_train["avg_flow_map"]
        self.norm_stats_target = target_info_from_train["norm_stats"]

    def __len__(self):
        return len(self.df_processed)

    def __getitem__(self, idx):
        target_key = (
            self.hour_category_for_target_grouping_np[idx], self.is_holiday_for_target_np[idx],
            self.s2_cond_category_for_target_np[idx], self.s3_cond_category_for_target_np[idx],
            self.s4_cond_category_for_target_np[idx]
        )
        target_avg_flow_np = self.average_flow_map_dict.get(target_key, np.zeros((self.H, self.W), dtype=np.float32))
        norm_target_np = (target_avg_flow_np - self.norm_stats_target['mean']) / self.norm_stats_target['std']
        target_tensor_norm = torch.from_numpy(norm_target_np).float().reshape(self.image_channels_target, self.D, self.H, self.W)

        condition_grids = []
        for col_name in self.feature_columns:
            val = self.original_values_dict[col_name][idx]
            stats = self.norm_stats_dict[col_name]
            norm_val = (val - stats['mean']) / stats['std'] if not np.isnan(val) else 0.0
            # 建立 (C=1, D=1, H, W) 的張量
            condition_grids.append(torch.full((1, self.D, self.H, self.W), float(norm_val), dtype=torch.float32))
        
        # DataLoader 會自動在 dim=0 加上批次維度
        condition_tensor_norm = torch.cat(condition_grids, dim=0) 

        return target_tensor_norm, condition_tensor_norm
        
# FID 函數 (get_activations, calculate_frechet_distance, calculate_fid)
def get_activations(images: torch.Tensor, model: nn.Module, device: str, batch_size_fid: int = 32) -> np.ndarray:
    """使用 Inception 模型提取影像特徵。"""
    model.eval()
    activations = []

    # 處理影像維度以符合 InceptionV3 輸入
    # images: (N, C, D, H, W)
    if images.shape[2] == 1: # D=1
        images_2d = images.squeeze(2) # (N, C, H, W)
    else: # D > 1, 取中間切片
        images_2d = images[:, :, images.shape[2]//2, :, :]
        logger.warning("影像深度 > 1，為 FID 取中間切片。")

    if images_2d.shape[1] == 1: # C=1, 複製為 3 通道
        images_2d = images_2d.repeat(1, 3, 1, 1)
    elif images_2d.shape[1] != 3 : # C != 1 且 C != 3, 取前 3 通道
        images_2d = images_2d[:,:3,:,:]
        logger.warning("影像通道數 != 1 或 3，為 FID 取前三通道。")

    # InceptionV3 需要 299x299 輸入
    transform_inception = transforms.Compose([
        transforms.Resize((299,299), antialias=True) # antialias=True 建議用於 PyTorch 1.7+
    ])

    num_batches = math.ceil(images_2d.shape[0] / batch_size_fid)
    for i in range(num_batches):
        batch = images_2d[i*batch_size_fid : (i+1)*batch_size_fid].to(device)
        batch = transform_inception(batch)
        with torch.no_grad():
            pred = model(batch) # InceptionV3 輸出
        if isinstance(pred, tuple): pred = pred[0] # 處理 InceptionV3 (非 aux_logits) 的輸出
        activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)


def calculate_frechet_distance(mu1:np.ndarray, sigma1:np.ndarray, mu2:np.ndarray, sigma2:np.ndarray, eps:float=1e-6) -> float:
    """計算兩個多元高斯分佈之間的 Fréchet Distance。"""
    mu1,mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1,sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "均值向量的形狀必須匹配"
    assert sigma1.shape == sigma2.shape, "共變異數矩陣的形狀必須匹配"

    diff = mu1 - mu2
    # 計算 (sigma1 * sigma2) 的平方根
    covmean_sqrt, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False) # disp=False 避免印出警告
    if not np.isfinite(covmean_sqrt).all(): # 處理數值不穩定
        offset = np.eye(sigma1.shape[0]) * eps
        covmean_sqrt = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean_sqrt): # 若結果是複數，取實部 (理論上應為實數)
        covmean_sqrt = covmean_sqrt.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean_sqrt)

def calculate_fid(real_acts:np.ndarray, gen_acts:np.ndarray)->float:
    """計算給定真實與生成影像特徵的 FID 分數。"""
    mu_real, sigma_real = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu_gen, sigma_gen = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

# Cell: 評估與視覺化函數 
def visualize_predictions_long_term(
                        generated_all_denorm_t: torch.Tensor, # (N, C, D, H, W) 反正規化後的生成數據
                        original_all_denorm_t: torch.Tensor,  # (N, C, D, H, W) 反正規化後的真實數據
                        config: Dict[str, Any],
                        sample_idx_to_plot: Optional[int] = 0, # 要繪製的特定樣本索引，None 表示繪製平均值
                        prefix: str = "test_eval", # 檔名前綴
                        grid_mask_hw: Optional[np.ndarray] = None # 【修正並新增參數】
                       ):
    """
    視覺化預測結果與真實值的比較 (針對 DDPM_Long-term.ipynb 的數據結構)。

    包含生成結果、真實數據、以及誤差（MSE、MAE、MAPE、SMAPE）的網格熱力圖。
    """
    save_dir = config.get("stage4_model_save_dir", config["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    if generated_all_denorm_t.shape[2] > 1 or original_all_denorm_t.shape[2] > 1:
        logger.warning(f"visualize_predictions_long_term: 數據深度 > 1，將取 D 維度的平均值進行繪圖。")
        generated_all_denorm_t = torch.mean(generated_all_denorm_t, dim=2, keepdim=True)
        original_all_denorm_t = torch.mean(original_all_denorm_t, dim=2, keepdim=True)

    generated_squeezed = generated_all_denorm_t.squeeze(1).squeeze(1) # (N, H, W)
    original_squeezed = original_all_denorm_t.squeeze(1).squeeze(1)   # (N, H, W)

    H, W = generated_squeezed.shape[-2], generated_squeezed.shape[-1]

    if sample_idx_to_plot is None:
        gen_data_to_plot = torch.mean(generated_squeezed, dim=0).cpu().numpy() # (H, W)
        orig_data_to_plot = torch.mean(original_squeezed, dim=0).cpu().numpy() # (H, W)
        title_suffix = "all_samples_avg"
    elif sample_idx_to_plot < generated_squeezed.shape[0]:
        gen_data_to_plot = generated_squeezed[sample_idx_to_plot].cpu().numpy()
        orig_data_to_plot = original_squeezed[sample_idx_to_plot].cpu().numpy()
        title_suffix = f"sample_{sample_idx_to_plot}"
    else:
        logger.warning(f"sample_idx_to_plot {sample_idx_to_plot} 超出範圍，將繪製平均值。")
        gen_data_to_plot = torch.mean(generated_squeezed, dim=0).cpu().numpy()
        orig_data_to_plot = torch.mean(original_squeezed, dim=0).cpu().numpy()
        title_suffix = "all_samples_avg_fallback"
    if grid_mask_hw is not None:
        # 確保遮罩是布林類型
        grid_mask_hw = grid_mask_hw.astype(bool)
        gen_data_to_plot[~grid_mask_hw] = np.nan
        orig_data_to_plot[~grid_mask_hw] = np.nan
    epsilon = 1e-8 # 避免除以零
    mse_matrix = (gen_data_to_plot - orig_data_to_plot) ** 2
    mae_matrix = np.abs(gen_data_to_plot - orig_data_to_plot)
    mape_matrix = np.abs((orig_data_to_plot - gen_data_to_plot) / (np.abs(orig_data_to_plot) + epsilon)) * 100
    smape_matrix = np.abs(gen_data_to_plot - orig_data_to_plot) / ((np.abs(orig_data_to_plot) + np.abs(gen_data_to_plot))/2 + epsilon) * 100 

    overall_mse = np.nanmean(mse_matrix)
    overall_mae = np.nanmean(mae_matrix)
    overall_mape = np.nanmean(mape_matrix[np.isfinite(mape_matrix)])
    overall_smape = np.nanmean(smape_matrix[np.isfinite(smape_matrix)])

    # --- 修改開始：繪製6個子圖 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # 改成 2x3 的佈局

    # 圖 1: Generated
    im_gen = axes[0, 0].imshow(gen_data_to_plot, cmap='viridis')
    axes[0, 0].set_title(f'Generated ({title_suffix})')
    axes[0, 0].axis('off') # 隱藏座標軸
    fig.colorbar(im_gen, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 圖 2: Original
    im_orig = axes[0, 1].imshow(orig_data_to_plot, cmap='viridis')
    axes[0, 1].set_title(f'Original ({title_suffix})')
    axes[0, 1].axis('off')
    fig.colorbar(im_orig, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 圖 3: MSE
    im_mse = axes[0, 2].imshow(mse_matrix, cmap='hot')
    axes[0, 2].set_title(f'MSE Grid (Avg: {overall_mse:.0f})')
    axes[0, 2].axis('off')
    fig.colorbar(im_mse, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 圖 4: MAE
    im_mae = axes[1, 0].imshow(mae_matrix, cmap='hot')
    axes[1, 0].set_title(f'MAE Grid (Avg: {overall_mae:.0f})')
    axes[1, 0].axis('off')
    fig.colorbar(im_mae, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 圖 5: MAPE
    # MAPE 值可能差異很大，可以考慮使用 vmin 和 vmax 來設定顯示範圍
    vmax_mape = np.percentile(mape_matrix[np.isfinite(mape_matrix)], 98) if np.any(np.isfinite(mape_matrix)) else 100 # 取98百分位數作為上限，避免極端值影響
    im_mape = axes[1, 1].imshow(mape_matrix, cmap='cividis', vmin=0, vmax=vmax_mape if vmax_mape > 0 else 100)
    axes[1, 1].set_title(f'MAPE Grid (Avg: {overall_mape:.0f})')
    axes[1, 1].axis('off')
    fig.colorbar(im_mape, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 圖 6: SMAPE
    # SMAPE 值通常在 0-200% 或 0-100% (取決於定義)
    vmax_smape = np.percentile(smape_matrix[np.isfinite(smape_matrix)], 98) if np.any(np.isfinite(smape_matrix)) else 100
    im_smape = axes[1, 2].imshow(smape_matrix, cmap='cividis', vmin=0, vmax=vmax_smape if vmax_smape > 0 else 100) # SMAPE 範圍 0-100% 或 0-200%
    axes[1, 2].set_title(f'SMAPE Grid (Avg: {overall_smape:.0f})')
    axes[1, 2].axis('off')
    fig.colorbar(im_smape, ax=axes[1, 2], fraction=0.046, pad=0.04)
  

    plt.tight_layout()
    # 更改儲存的檔名以反映是六張圖的比較
    plt.savefig(os.path.join(save_dir, f'{prefix}_6maps_comparison_{title_suffix}.png'), dpi=300)
    plt.close(fig)

def plot_grid_with_error_long_term(
                        dataset_for_coords: Any, # 實際應為 Stage2Dataset 或類似結構
                        error_metrics_grids: Dict[str, np.ndarray],
                        config: Dict[str, Any],
                        prefix: str = "test_eval",
                        grid_mask_flat_indices: Optional[np.ndarray] = None
                       ):
    logger_func = logging.getLogger(__name__) # 確保 logger 在函數作用域內可用
    save_dir = config.get("stage4_model_save_dir", config.get("save_dir")) 
    os.makedirs(save_dir, exist_ok=True)

    H, W = config["H"], config["W"]

    # 從 config 中獲取網格映射信息
    sorted_flow_columns_map = config.get("cached_basemodel_sorted_flow_columns")
    grid_idx_to_rc_map_plot = config.get("cached_basemodel_grid_idx_to_rc_map")
    selected_sensor_info_plot = config.get("cached_basemodel_selected_sensor_info")

    if not all([sorted_flow_columns_map, grid_idx_to_rc_map_plot, selected_sensor_info_plot]):
        logger_func.error("plot_grid_with_error_long_term: CONFIG 中缺少必要的網格映射資訊。")
        return

    selected_sensor_info_dict = {info['name']: (info['lon'], info['lat'])
                             for info in selected_sensor_info_plot if isinstance(info, dict) and 'name' in info}

    all_sensor_lons, all_sensor_lats = [], []
    for flat_grid_idx in range(H * W):
        lon, lat = np.nan, np.nan
        if flat_grid_idx < len(sorted_flow_columns_map):
            col_name = sorted_flow_columns_map[flat_grid_idx]
            if col_name in selected_sensor_info_dict:
                lon, lat = selected_sensor_info_dict[col_name]
        all_sensor_lons.append(lon)
        all_sensor_lats.append(lat)

    all_sensor_lons = np.array(all_sensor_lons)
    all_sensor_lats = np.array(all_sensor_lats)

    # 【新增邏輯】如果提供了遮罩，則只使用遮罩內的座標和誤差值
    if grid_mask_flat_indices is not None:
        actual_sensor_lons = all_sensor_lons[grid_mask_flat_indices]
        actual_sensor_lats = all_sensor_lats[grid_mask_flat_indices]
        valid_grid_indices_flat = grid_mask_flat_indices
    else:
        actual_sensor_lons = all_sensor_lons
        actual_sensor_lats = all_sensor_lats
        valid_grid_indices_flat = np.arange(H * W)

    # 移除經緯度為 NaN 的點，避免繪圖錯誤
    valid_coords_mask = ~np.isnan(actual_sensor_lons) & ~np.isnan(actual_sensor_lats)
    actual_sensor_lons = actual_sensor_lons[valid_coords_mask]
    actual_sensor_lats = actual_sensor_lats[valid_coords_mask]
    # 同樣篩選 valid_grid_indices_flat，確保誤差值與座標對應
    valid_grid_indices_flat = valid_grid_indices_flat[valid_coords_mask]


    if len(actual_sensor_lons) == 0:
            logger_func.error("plot_grid_with_error: 無法獲取任何網格點的座標。")
            return

    # 1. 原始的「黑到紅」色票，用於絕對誤差 (MSE, MAE...)
    cdict_black_to_red = {
        'red':   ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    black_to_red_cmap = mcolors.LinearSegmentedColormap('BlackToRed', cdict_black_to_red)

    # 2. 新的「紅-黑-綠」發散色票，用於差異圖 (Difference)
    #    紅色代表負數，綠色代表正數，中心為黑色
    cdict_div_RedBlackGreen = {
        'red':   ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
        'blue':  ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    RedBlackGreen_div_cmap = mcolors.LinearSegmentedColormap('RedBlackGreenDiv', cdict_div_RedBlackGreen)

    # --- 修改結束 ---


    for metric_name, error_grid_flat in error_metrics_grids.items():
        if not isinstance(error_grid_flat, np.ndarray) or error_grid_flat.ndim == 0 or error_grid_flat.shape[0] != H*W :
            logger_func.error(f"指標 {metric_name} 的誤差網格維度不正確。跳過繪圖。")
            continue

        error_values_for_plot = error_grid_flat[valid_grid_indices_flat]
        
        if len(error_values_for_plot) == 0:
            logger_func.warning(f"沒有可用於繪圖的有效誤差值：{metric_name}。跳過繪圖。")
            continue
        
        error_values_for_plot_display = np.where(np.isfinite(error_values_for_plot), error_values_for_plot, np.nan)

        is_diff_plot = 'diff' in metric_name.lower()

        if is_diff_plot:
            title_to_display = f"Geographic Grid Error Difference - {metric_name} ({prefix})"
            
            vmin = np.nanmin(error_values_for_plot_display)
            vmax = np.nanmax(error_values_for_plot_display)

            if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
                # 邊界情況：如果數據無效或全為同一個值，給定一個預設範圍
                vmin, vmax = -1.0, 1.0
                norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
                cmap_to_use = RedBlackGreen_div_cmap
            elif vmin < 0 and vmax > 0:
                # 情況1: 數據跨越 0 (有正有負)，使用非對稱的 TwoSlopeNorm
                norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
                cmap_to_use = RedBlackGreen_div_cmap
            elif vmax <= 0:
                # 情況2: 數據全部為負或零，只使用「紅到黑」部分
                norm = mcolors.Normalize(vmin=vmin, vmax=0)
                # 從完整的發散色票中，只截取代表負值的前半部分 (0.0 到 0.5)
                cmap_to_use = mcolors.LinearSegmentedColormap.from_list(
                    'RedToBlack_trunc', RedBlackGreen_div_cmap(np.linspace(0, 0.5, 128))
                )
            else: # vmin >= 0
                # 情況3: 數據全部為正或零，只使用「黑到綠」部分
                norm = mcolors.Normalize(vmin=0, vmax=vmax)
                # 從完整的發散色票中，只截取代表正值的後半部分 (0.5 到 1.0)
                cmap_to_use = mcolors.LinearSegmentedColormap.from_list(
                    'BlackToGreen_trunc', RedBlackGreen_div_cmap(np.linspace(0.5, 1.0, 128))
                )
            
            # 將 norm 和 cmap 傳遞給繪圖指令
            scatter_args = {'cmap': cmap_to_use, 'norm': norm}

        else: # 非差異圖的邏輯保持不變
            title_to_display = f"Geographic Grid Error Heatmap - {metric_name.upper()} ({prefix})"
            min_val = np.nanmin(error_values_for_plot_display)
            max_val = np.nanmax(error_values_for_plot_display)
            if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
                min_val, max_val = 0.0, 1.0
            scatter_args = {'cmap': black_to_red_cmap, 'vmin': min_val, 'vmax': max_val}


        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(actual_sensor_lons, actual_sensor_lats, c=error_values_for_plot_display, 
                                marker='s', s=100, **scatter_args)
        
        plt.colorbar(scatter, label=metric_name)

        if metric_name.upper() != 'MSE':
            for i in range(len(actual_sensor_lons)):
                val_to_text = error_values_for_plot_display[i]
                if not np.isnan(val_to_text):
                    plt.text(actual_sensor_lons[i], actual_sensor_lats[i],
                             f'{val_to_text:.0f}',
                             fontsize=6, color='white', ha='center', va='center')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title_to_display)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(save_dir, f'{prefix}_grid_{metric_name.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger_func.info(f"已儲存 {metric_name} 的地理網格誤差圖: {prefix}")                        

def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

@torch.no_grad()
def evaluate_model(
    current_stage_model_trained: 'DDPM3D',
    previous_stage_model_eval_instance: Optional['DDPM3D'],
    basemodel_eval_instance_for_s2_cond_generation: Optional['DDPM3D'],
    current_stage_mode: ConditionMode,
    dataloader_current_stage: DataLoader,
    inception_model_fid: nn.Module,
    config: Dict[str, Any],
    current_stage_target_norm_stats: Dict[str, float],
    target_grid_stds: np.ndarray,
    target_overall_std: float,
    previous_stage_target_norm_stats: Optional[Dict[str, float]] = None,
    max_samples_for_fid: Optional[int] = None,
    prefix: str = "eval",
    grid_mask_hw: Optional[np.ndarray] = None, 
    grid_mask_flat_indices: Optional[np.ndarray] = None 
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:

    current_stage_name = current_stage_mode.name
    eval_type_log = "FILTERED GRID" if grid_mask_hw is not None else "FULL MAP"
    logger.info(f"===== 開始 {current_stage_name} 模型評估 (類型: {eval_type_log}, 比較: {prefix}) =====")
    if previous_stage_model_eval_instance:
        previous_stage_model_eval_instance.eval()
    if basemodel_eval_instance_for_s2_cond_generation and current_stage_mode == ConditionMode.STAGE3: # S3的prev是S2, S2的prev是BM
        basemodel_eval_instance_for_s2_cond_generation.eval()

    inception_model_fid.eval()

    # 獲取並驗證當前階段目標的反正規化統計量
    if not (current_stage_target_norm_stats and \
            'mean' in current_stage_target_norm_stats and \
            'std' in current_stage_target_norm_stats):
        logger.error(f"{current_stage_name} 評估: {current_stage_name} 目標的專用正規化統計量缺失或不完整。")
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape_avg_grid": float('nan'), "smape_avg_grid": float('nan'), "mape_overall": float('nan'), "smape_overall": float('nan'), "fid": float('nan')}
        nan_grids_dict = {m: np.array([np.nan]) for m in ['MSE', 'MAE', 'MAPE', 'SMAPE']}
        current_model_key_eval = f"{current_stage_name.lower()}_model"
        results_to_return = {current_model_key_eval: nan_metrics}
        error_grids_to_return = {current_model_key_eval: nan_grids_dict}
        if previous_stage_model_eval_instance:
            prev_model_key_eval = f"{ConditionMode(current_stage_mode.value -1).name.lower()}_model_on_{current_stage_name.lower()}_data"
            results_to_return[prev_model_key_eval] = nan_metrics
            error_grids_to_return[prev_model_key_eval] = nan_grids_dict
        return results_to_return, error_grids_to_return

    cs_target_mean_for_denorm = current_stage_target_norm_stats['mean']
    cs_target_std_for_denorm = current_stage_target_norm_stats['std']
    logger.info(f"{current_stage_name} 評估: 使用 {current_stage_name} 目標專用統計量進行反正規化: mean={cs_target_mean_for_denorm:.4f}, std={cs_target_std_for_denorm:.4f}")
    if cs_target_std_for_denorm < 1e-6:
        cs_target_std_for_denorm = 1.0
    
    ps_model_output_mean_for_denorm = cs_target_mean_for_denorm # 預設回退 (不應該發生)
    ps_model_output_std_for_denorm = cs_target_std_for_denorm   # 預設回退
    prev_stage_log_name = "N/A"
    if previous_stage_model_eval_instance:
        if current_stage_mode.value <= ConditionMode.BASEMODEL.value : # Should not happen if prev exists
            logger.error(f"無法為 {current_stage_name} 的前一階段確定模式。")
        else:
            prev_stage_log_name = ConditionMode(current_stage_mode.value - 1).name
            if previous_stage_target_norm_stats and 'mean' in previous_stage_target_norm_stats and 'std' in previous_stage_target_norm_stats:
                ps_model_output_mean_for_denorm = previous_stage_target_norm_stats['mean']
                ps_model_output_std_for_denorm = previous_stage_target_norm_stats['std']
                if ps_model_output_std_for_denorm < 1e-6:
                    ps_model_output_std_for_denorm = 1.0
                logger.info(f"將使用 {prev_stage_log_name} 目標的專用統計量反正規化其模型的輸出: mean={ps_model_output_mean_for_denorm:.4f}, std={ps_model_output_std_for_denorm:.4f}")
            else:
                logger.warning(f"未提供 {prev_stage_log_name} 目標的專用正規化統計量，其模型反正規化將使用 {current_stage_name} 目標統計量（可能不準確）。")

    max_fid_samples_actual = len(dataloader_current_stage.dataset)
    if max_samples_for_fid is not None:
        max_fid_samples_actual = min(max_samples_for_fid, max_fid_samples_actual)
    logger.info(f"將為 FID 計算收集最多 {max_fid_samples_actual} 個樣本。")

    # 初始化收集列表
    all_cs_generated_denorm_list: List[torch.Tensor] = []
    all_ps_generated_denorm_on_cs_data_list: List[torch.Tensor] = [] # Previous Stage
    all_cs_target_denorm_list: List[torch.Tensor] = []

    all_cs_generated_norm_for_fid_list: List[torch.Tensor] = []
    all_ps_generated_norm_for_fid_on_cs_data_list: List[torch.Tensor] = [] # Previous Stage
    all_cs_target_norm_for_fid_list: List[torch.Tensor] = []

    pbar_eval = tqdm(dataloader_current_stage, desc=f"{current_stage_name} 評估 ({prefix})", leave=False)
    for batch_idx, batch_data in enumerate(pbar_eval):
        # 根據 MultiStageDataset.__getitem__ 的返回順序解包
        # (target_cs, cond1_cs, cond2_cs, bm_hr, bm_hol, bm_out_for_s2, s2_new_for_s2, s2_out_for_s3, s3_new_for_s3, s3_out_for_s4, s4_new_for_s4, s2_orig, s3_orig, s4_orig)
        cs_target_eval_norm                       = batch_data[0].to(config["device"])
        cs_cond1_input_norm                     = batch_data[1].to(config["device"]) # 當前階段條件1 (前一階段輸出)
        cs_cond2_input_norm                     = batch_data[2].to(config["device"]) # 當前階段條件2 (當前階段新特徵)
        
        bm_hr_scalar_batch                      = batch_data[3].to(config["device"]) # BM原始小時
        bm_hol_scalar_batch                     = batch_data[4].to(config["device"]) # BM原始假日
        
        bm_out_grid_for_s2_cond_norm            = batch_data[5].to(config["device"]) # BM輸出網格 (S2條件1)
        s2_new_feat_grid_for_s2_cond_norm       = batch_data[6].to(config["device"]) # S2新特徵網格 (S2條件2)
        
        s2_out_grid_for_s3_cond_norm            = batch_data[7].to(config["device"]) # S2輸出網格 (S3條件1)
        s3_new_feat_grid_for_s3_cond_norm       = batch_data[8].to(config["device"]) # S3新特徵網格 (S3條件2)

        s3_out_grid_for_s4_cond_norm            = batch_data[9].to(config["device"]) # S3輸出網格 (S4條件1)
        s4_new_feat_grid_for_s4_cond_norm       = batch_data[10].to(config["device"])# S4新條件網格 (S4條件2)
        
        # s2_original_feature_scalar_batch          = batch_data[11].to(config["device"]) # 未在本函數直接使用
        # s3_original_feature_scalar_batch          = batch_data[12].to(config["device"]) # 未在本函數直接使用
        # s4_original_feature_scalar_batch          = batch_data[13].to(config["device"]) # 未在本函數直接使用

        current_batch_size = cs_target_eval_norm.shape[0]

        # --- 1. 當前階段模型生成與反正規化 ---
        cs_model_cond_args = {}
        if current_stage_mode == ConditionMode.STAGE2: # 雖然此函數主要用於S3/S4，但為保持通用性
            cs_model_cond_args = {"basemodel_output_grid_batch": cs_cond1_input_norm, 
                                  "stage2_new_condition_feature_grid_batch": cs_cond2_input_norm}
        elif current_stage_mode == ConditionMode.STAGE3:
            cs_model_cond_args = {"stage2_output_grid_batch_for_s3": cs_cond1_input_norm, 
                                  "stage3_new_condition_feature_grid_batch": cs_cond2_input_norm}
        elif current_stage_mode == ConditionMode.STAGE4:
            cs_model_cond_args = {"stage3_output_grid_batch_for_s4": cs_cond1_input_norm, 
                                  "stage4_new_condition_feature_grid_batch": cs_cond2_input_norm}
        
        cs_generated_eval_norm = current_stage_model_trained.sample(current_batch_size, current_stage_mode, cs_model_cond_args)
        cs_generated_eval_denorm = cs_generated_eval_norm * cs_target_std_for_denorm + cs_target_mean_for_denorm
        cs_generated_eval_denorm = torch.clamp(cs_generated_eval_denorm, min=0.0)
        all_cs_generated_denorm_list.append(cs_generated_eval_denorm.cpu())
        all_cs_generated_norm_for_fid_list.append(cs_generated_eval_norm.cpu())

        # --- 2. 前一階段模型生成與反正規化 (如果存在) ---
        if previous_stage_model_eval_instance:
            prev_stage_mode_to_eval = ConditionMode(current_stage_mode.value - 1)
            ps_model_cond_args = {}
            if prev_stage_mode_to_eval == ConditionMode.BASEMODEL:
                ps_model_cond_args = {"hour_scalars_batch": bm_hr_scalar_batch, 
                                      "is_holiday_scalars_batch": bm_hol_scalar_batch}
            elif prev_stage_mode_to_eval == ConditionMode.STAGE2:
                ps_model_cond_args = {"basemodel_output_grid_batch": bm_out_grid_for_s2_cond_norm, 
                                      "stage2_new_condition_feature_grid_batch": s2_new_feat_grid_for_s2_cond_norm}
            elif prev_stage_mode_to_eval == ConditionMode.STAGE3:
                ps_model_cond_args = {"stage2_output_grid_batch_for_s3": s2_out_grid_for_s3_cond_norm, 
                                      "stage3_new_condition_feature_grid_batch": s3_new_feat_grid_for_s3_cond_norm}
            
            if ps_model_cond_args: # 確保條件準備好了
                ps_generated_eval_norm = previous_stage_model_eval_instance.sample(current_batch_size, prev_stage_mode_to_eval, ps_model_cond_args)
                
                # 使用前一階段自身的目標統計量進行反正規化
                ps_generated_eval_denorm = ps_generated_eval_norm * ps_model_output_std_for_denorm + ps_model_output_mean_for_denorm
                ps_generated_eval_denorm = torch.clamp(ps_generated_eval_denorm, min=0.0)
                all_ps_generated_denorm_on_cs_data_list.append(ps_generated_eval_denorm.cpu())
                all_ps_generated_norm_for_fid_on_cs_data_list.append(ps_generated_eval_norm.cpu())

                logger.info(f"{current_stage_name} Eval - Batch {batch_idx + 1} - RAW {prev_stage_mode_to_eval.name} NORM Output Stats: "
                            f"Min: {torch.min(ps_generated_eval_norm).item():.4f}, Max: {torch.max(ps_generated_eval_norm).item():.4f}, "
                            f"Mean: {torch.mean(ps_generated_eval_norm).item():.4f}, Std: {torch.std(ps_generated_eval_norm).item():.4f}")
            else:
                logger.warning(f"無法為前一階段 ({prev_stage_mode_to_eval.name}) 準備有效的 sample 條件，跳過其在此批次的生成。")
        
        # --- 當前階段的目標反正規化 ---
        cs_target_eval_denorm = cs_target_eval_norm * cs_target_std_for_denorm + cs_target_mean_for_denorm
        all_cs_target_denorm_list.append(cs_target_eval_denorm.cpu())

        # --- 為 FID 收集正規化樣本 ---
        samples_collected_so_far = sum(s.shape[0] for s in all_cs_target_norm_for_fid_list)
        if samples_collected_so_far < max_fid_samples_actual:
            remaining_needed_fid = max_fid_samples_actual - samples_collected_so_far
            samples_to_add_fid = min(current_batch_size, remaining_needed_fid)
            if samples_to_add_fid > 0:
                all_cs_target_norm_for_fid_list.append(cs_target_eval_norm[:samples_to_add_fid].cpu())
                all_cs_generated_norm_for_fid_list[-1] = all_cs_generated_norm_for_fid_list[-1][:samples_to_add_fid] # Truncate
                if previous_stage_model_eval_instance and all_ps_generated_norm_for_fid_on_cs_data_list:
                     all_ps_generated_norm_for_fid_on_cs_data_list[-1] = all_ps_generated_norm_for_fid_on_cs_data_list[-1][:samples_to_add_fid] # Truncate
    
    # --- 迴圈結束後，匯總和計算指標 ---
    if not all_cs_target_denorm_list:
        logger.warning(f"{current_stage_name} 評估 ({prefix}): 無數據處理或收集到。")
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape_avg_grid": float('nan'), "smape_avg_grid": float('nan'), "mape_overall": float('nan'), "smape_overall": float('nan'), "fid": float('nan')}
        nan_grids_dict = {m: np.array([np.nan]) for m in ['MSE', 'MAE', 'MAPE', 'SMAPE']}
        current_model_key_eval = f"{current_stage_name.lower()}_model"
        results_to_return = {current_model_key_eval: nan_metrics}
        error_grids_to_return = {current_model_key_eval: nan_grids_dict}
        if previous_stage_model_eval_instance:
            prev_model_key_eval = f"{ConditionMode(current_stage_mode.value -1).name.lower()}_model_on_{current_stage_name.lower()}_data"
            results_to_return[prev_model_key_eval] = nan_metrics
            error_grids_to_return[prev_model_key_eval] = nan_grids_dict
        return results_to_return, error_grids_to_return

    cs_target_all_t = torch.cat(all_cs_target_denorm_list, dim=0)
    
    results = {}
    error_grids_all_models: Dict[str, Dict[str, Any]] = {}
    epsilon = 1e-8

    model_predictions_map_eval = {}
    # 當前階段模型
    cs_model_key_name = f"{current_stage_name.lower()}_model"
    if all_cs_generated_denorm_list:
        cs_generated_all_t = torch.cat(all_cs_generated_denorm_list, dim=0)
        logger.info(f"{cs_model_key_name} (反正規化後) shape: {cs_generated_all_t.shape}, "
                    f"min: {torch.min(cs_generated_all_t).item():.4f}, max: {torch.max(cs_generated_all_t).item():.4f}, mean: {torch.mean(cs_generated_all_t).item():.4f}, std: {torch.std(cs_generated_all_t).item():.4f}")
        model_predictions_map_eval[cs_model_key_name] = (cs_generated_all_t, all_cs_generated_norm_for_fid_list)

    # 前一階段模型
    if previous_stage_model_eval_instance and all_ps_generated_denorm_on_cs_data_list:
        prev_stage_mode_eval = ConditionMode(current_stage_mode.value - 1)
        ps_model_key_name = f"{prev_stage_mode_eval.name.lower()}_model_on_{current_stage_name.lower()}_data"
        ps_generated_all_t = torch.cat(all_ps_generated_denorm_on_cs_data_list, dim=0)
        logger.info(f"{ps_model_key_name} (反正規化後) shape: {ps_generated_all_t.shape}, "
                    f"min: {torch.min(ps_generated_all_t).item():.4f}, max: {torch.max(ps_generated_all_t).item():.4f}, mean: {torch.mean(ps_generated_all_t).item():.4f}, std: {torch.std(ps_generated_all_t).item():.4f}")
        model_predictions_map_eval[ps_model_key_name] = (ps_generated_all_t, all_ps_generated_norm_for_fid_on_cs_data_list)
    mask_tensor = None
    if grid_mask_hw is not None:
        mask_tensor = torch.from_numpy(grid_mask_hw.astype(bool)).to(cs_target_all_t.device)
        # 擴展到與數據相同的維度: (N, C, D, H, W)
        mask_tensor = mask_tensor.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(cs_target_all_t)

    for model_name, (pred_t, gen_fid_list_for_model) in model_predictions_map_eval.items():
        pred_for_metric = pred_t
        target_for_metric = cs_target_all_t
        if mask_tensor is not None:
            # torch.masked_select 會返回一個 1D 張量，這對於計算純量指標是OK的
            pred_for_metric = torch.masked_select(pred_t, mask_tensor)
            target_for_metric = torch.masked_select(cs_target_all_t, mask_tensor)

        mse = F.mse_loss(pred_for_metric, target_for_metric).item()
        mae = F.l1_loss(pred_for_metric, target_for_metric).item()

        actual_values_for_mape = torch.abs(target_for_metric)
        errors_for_mape = torch.abs(target_for_metric - pred_for_metric)
        threshold_for_mape_calc = config.get("mape_threshold", 1.0) 
        valid_for_mape_calc_mask = actual_values_for_mape > threshold_for_mape_calc
        if torch.sum(valid_for_mape_calc_mask).item() > 0:
            mape_per_element_filtered = (errors_for_mape[valid_for_mape_calc_mask] / actual_values_for_mape[valid_for_mape_calc_mask]) * 100
            mape_avg_grid = torch.mean(mape_per_element_filtered[torch.isfinite(mape_per_element_filtered)]).item()
        else:
            mape_avg_grid = float('inf')

        smape_numerator = torch.abs(pred_for_metric - target_for_metric)
        smape_denominator = (torch.abs(target_for_metric) + torch.abs(pred_for_metric)) / 2.0 + epsilon
        smape_tensor = (smape_numerator / smape_denominator) * 100
        smape_avg_grid = torch.mean(smape_tensor[torch.isfinite(smape_tensor)]).item()

        mape_overall_numerator = torch.sum(errors_for_mape)
        mape_overall_denominator = torch.sum(actual_values_for_mape) + epsilon
        mape_overall = (mape_overall_numerator / mape_overall_denominator).item() * 100

        smape_overall_denominator_sum_abs = torch.sum(torch.abs(target_for_metric) + torch.abs(pred_for_metric))
        smape_overall = (200.0 * mape_overall_numerator / (smape_overall_denominator_sum_abs + epsilon)).item()
        grid_stds_t = torch.from_numpy(target_grid_stds.reshape(1, 1, 1, config["H"], config["W"])).to(pred_t.device)
        standardized_error_map_t = torch.mean(torch.abs(pred_t - cs_target_all_t) / (grid_stds_t + epsilon), dim=0).squeeze()
        stde_g = standardized_error_map_t.cpu().numpy().flatten()

        # 再根據遮罩計算 STDE_avg_grid
        if grid_mask_flat_indices is not None:
            stde_avg_grid = np.nanmean(stde_g[grid_mask_flat_indices])
            # mae 是在篩選後數據上計算的，所以這裡的 std 也應該在篩選後數據上計算
            target_overall_std_filtered = torch.std(target_for_metric).item()
            stde_overall = mae / (target_overall_std_filtered + epsilon)
        else:
            stde_avg_grid = np.nanmean(stde_g)
            stde_overall = mae / (target_overall_std + epsilon) # 全地圖使用傳入的全域 std
        
        fid = float('nan')
        if gen_fid_list_for_model and all_cs_target_norm_for_fid_list: # 使用當前模型的 FID 列表
            if not all(len(lst) > 0 for lst in [gen_fid_list_for_model, all_cs_target_norm_for_fid_list]):
                 logger.warning(f"FID for {model_name}: Not enough batches collected for FID sample lists.")
            else:
                gen_fid_tensor = torch.cat(gen_fid_list_for_model, dim=0)[:max_fid_samples_actual]
                real_fid_tensor = torch.cat(all_cs_target_norm_for_fid_list, dim=0)[:max_fid_samples_actual]
                num_fid = min(gen_fid_tensor.shape[0], real_fid_tensor.shape[0])
                if num_fid > 1:
                    logger.info(f"Calculating FID for {model_name} (vs {current_stage_name} target) on {num_fid} samples...")
                    try:
                        act_gen = get_activations(gen_fid_tensor, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                        act_real = get_activations(real_fid_tensor, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                        if act_gen.shape[0] > 1 and act_real.shape[0] > 1:
                            fid = calculate_fid(act_real, act_gen)
                        else: logger.warning(f"FID for {model_name}: Insufficient features after activations.")
                    except Exception as e_fid: logger.error(f"FID calculation for {model_name} failed: {e_fid}")
                else: logger.warning(f"FID for {model_name}: Insufficient samples after concatenation ({num_fid}).")
        else: logger.warning(f"FID for {model_name}: FID sample lists were empty.")

        results[model_name] = {
            "mse": mse, "mae": mae, 
            "mape_avg_grid": mape_avg_grid, "smape_avg_grid": smape_avg_grid,
            "mape_overall": mape_overall, "smape_overall": smape_overall,
            "stde_avg_grid": stde_avg_grid,
            "stde_overall": stde_overall,
            "fid": fid if np.isfinite(fid) else float('nan')
        }
        logger.info(f"Metrics for {model_name} ({prefix}): {results[model_name]}")

        # 計算逐網格誤差
        if pred_t.ndim == 5 and pred_t.shape[1] == config["image_channels"] and pred_t.shape[2:] == (config.get("D",1), config["H"], config["W"]):
            pred_squeezed = pred_t.squeeze(1).squeeze(1) 
            target_squeezed = cs_target_all_t.squeeze(1).squeeze(1)
            mse_g = torch.mean((pred_squeezed - target_squeezed)**2, dim=0).cpu().numpy()
            mae_g = torch.mean(torch.abs(pred_squeezed - target_squeezed), dim=0).cpu().numpy()
            mape_g_t = torch.abs((target_squeezed - pred_squeezed) / (torch.abs(target_squeezed) + epsilon)) * 100
            mape_g = torch.mean(mape_g_t, dim=0).cpu().numpy()
            smape_n_g = torch.abs(pred_squeezed - target_squeezed)
            smape_d_g = (torch.abs(target_squeezed) + torch.abs(pred_squeezed))/2.0 + epsilon
            smape_g_t = (smape_n_g / smape_d_g) * 100
            smape_g = torch.mean(smape_g_t, dim=0).cpu().numpy()
            error_grids_all_models[model_name] = {
                'MSE': mse_g.flatten(), 'MAE': mae_g.flatten(),
                'MAPE': mape_g.flatten(), 'SMAPE': smape_g.flatten(),
                'STDE_AvgGrid': stde_g
            }
        else:
            error_grids_all_models[model_name] = {m: np.full((config["H"] * config["W"],), np.nan) for m in ['MSE','MAE','MAPE','SMAPE']}

    logger.info(f"Generating visualizations for {current_stage_name} evaluation ({prefix})...")
    dataset_obj_for_viz = dataloader_current_stage.dataset # 用於傳遞給繪圖函數
    
    # 視覺化當前階段模型
    cs_model_key_viz = f"{current_stage_name.lower()}_model"
    if cs_model_key_viz in model_predictions_map_eval:
        cs_pred_tensor_viz, _ = model_predictions_map_eval[cs_model_key_viz]
        if cs_pred_tensor_viz.shape[0] > 0 and cs_target_all_t.shape[0] > 0:
            visualize_predictions_long_term(
                cs_pred_tensor_viz[0:1].clone().cpu(), cs_target_all_t[0:1].clone().cpu(),
                config, sample_idx_to_plot=0,
                prefix=f"{prefix}_{cs_model_key_viz}_vs_{current_stage_name}Target_sample0",
                grid_mask_hw=grid_mask_hw
            )
            visualize_predictions_long_term(
                torch.mean(cs_pred_tensor_viz, dim=0, keepdim=True).clone().cpu(),
                torch.mean(cs_target_all_t, dim=0, keepdim=True).clone().cpu(),
                config, sample_idx_to_plot=None,
                prefix=f"{prefix}_{cs_model_key_viz}_vs_{current_stage_name}Target_avg",
                grid_mask_hw=grid_mask_hw
            )
        if cs_model_key_viz in error_grids_all_models:
            plot_grid_with_error_long_term(
                dataset_obj_for_viz, error_grids_all_models[cs_model_key_viz], config,
                f"{prefix}_{cs_model_key_viz}", grid_mask_flat_indices=grid_mask_flat_indices
            )
    
    # 視覺化前一階段模型
    if previous_stage_model_eval_instance:
        prev_stage_enum_for_viz = ConditionMode(current_stage_mode.value - 1)
        ps_model_key_viz = f"{prev_stage_enum_for_viz.name.lower()}_model_on_{current_stage_name.lower()}_data"
        if ps_model_key_viz in model_predictions_map_eval:
            ps_pred_tensor_viz, _ = model_predictions_map_eval[ps_model_key_viz]
            if ps_pred_tensor_viz.shape[0] > 0 and cs_target_all_t.shape[0] > 0:
                visualize_predictions_long_term(
                    ps_pred_tensor_viz[0:1].clone().cpu(), cs_target_all_t[0:1].clone().cpu(),
                    config, sample_idx_to_plot=0,
                    prefix=f"{prefix}_{prev_stage_enum_for_viz.name}_vs_{current_stage_name}Target_sample0",
                    grid_mask_hw=grid_mask_hw
                )
                visualize_predictions_long_term(
                    torch.mean(ps_pred_tensor_viz, dim=0, keepdim=True).clone().cpu(),
                    torch.mean(cs_target_all_t, dim=0, keepdim=True).clone().cpu(),
                    config, sample_idx_to_plot=None,
                    prefix=f"{prefix}_{prev_stage_enum_for_viz.name}_vs_{current_stage_name}Target_avg",
                    grid_mask_hw=grid_mask_hw
                )
            if ps_model_key_viz in error_grids_all_models:
                plot_grid_with_error_long_term(
                    dataset_obj_for_viz, error_grids_all_models[ps_model_key_viz], config,
                    f"{prefix}_{prev_stage_enum_for_viz.name.lower()}", grid_mask_flat_indices=grid_mask_flat_indices
                )

    # 計算並繪製誤差差異圖 (當前階段 vs 前一階段)
    if previous_stage_model_eval_instance:
        prev_stage_enum_for_diff = ConditionMode(current_stage_mode.value - 1)
        cs_model_key_for_diff = f"{current_stage_name.lower()}_model"
        ps_model_key_for_diff = f"{prev_stage_enum_for_diff.name.lower()}_model_on_{current_stage_name.lower()}_data"

        if cs_model_key_for_diff in error_grids_all_models and ps_model_key_for_diff in error_grids_all_models:
            cs_err = error_grids_all_models[cs_model_key_for_diff]
            ps_err = error_grids_all_models[ps_model_key_for_diff]
            diff_cs_ps_grids = {}
            for metric_key_diff in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
                if metric_key_diff in cs_err and isinstance(cs_err[metric_key_diff], np.ndarray) and \
                   metric_key_diff in ps_err and isinstance(ps_err[metric_key_diff], np.ndarray) and \
                   cs_err[metric_key_diff].shape == ps_err[metric_key_diff].shape:
                    diff_cs_ps_grids[f"Diff_{metric_key_diff}_({current_stage_name}-{prev_stage_enum_for_diff.name})"] = cs_err[metric_key_diff] - ps_err[metric_key_diff]
            if diff_cs_ps_grids:
                plot_grid_with_error_long_term(
                    dataset_obj_for_viz, diff_cs_ps_grids, config, 
                    f"{prefix}_diff_{current_stage_name}_minus_{prev_stage_enum_for_diff.name}",
                    grid_mask_flat_indices=grid_mask_flat_indices
                )
            
    # --- Excel 匯出 ---
    excel_rows_to_export = []
    logger.info(f"{current_stage_name} 評估 ({prefix}, {eval_type_log}): 開始準備 Excel 報告的詳細指標...")
    num_grid_cells_eval = config["H"] * config["W"]
    grid_idx_to_rc_map_excel = config.get("cached_basemodel_grid_idx_to_rc_map")
    sorted_flow_columns_excel = config.get("cached_basemodel_sorted_flow_columns")
    selected_sensor_info_excel = config.get("cached_basemodel_selected_sensor_info")
    sensor_info_lookup_excel = {}
    if selected_sensor_info_excel:
        sensor_info_lookup_excel = {
            info['name']: {'lon': info['lon'], 'lat': info['lat']}
            for info in selected_sensor_info_excel if isinstance(info, dict) and 'name' in info
        }
    if not all([grid_idx_to_rc_map_excel, sorted_flow_columns_excel, selected_sensor_info_excel]):
        logger.error(f"{current_stage_name} 評估 ({prefix}): Excel 報告缺少網格映射資訊。")

    for model_key_excel in model_predictions_map_eval.keys():
        if model_key_excel not in results or model_key_excel not in error_grids_all_models:
            logger.warning(f"Excel: 模型 {model_key_excel} 結果缺失。")
            continue
        metrics_eval_excel = results[model_key_excel]
        error_grids_eval_excel = error_grids_all_models[model_key_excel]

        excel_rows_to_export.append({'資料來源': f"--- {model_key_excel} ({prefix} vs {current_stage_name} Target) ---",
                                    '網格座標_R': '', '網格座標_C': '', '經度': '', '緯度': '', 'MSE': '', 'MAE': '', 
                                    'MAPE (AvgGrid)': '', 'SMAPE (AvgGrid)': '', 'MAPE (Overall)': '', 'SMAPE (Overall)': '', 'FID': '', 'STDE_AvgGrid': ''})

        indices_to_loop = grid_mask_flat_indices if grid_mask_flat_indices is not None else range(num_grid_cells_eval)

        for flat_idx in indices_to_loop:
            grid_r_coord_excel, grid_c_coord_excel = ('N/A', 'N/A') if not grid_idx_to_rc_map_excel else grid_idx_to_rc_map_excel.get(flat_idx, ('N/A','N/A'))
            lon_coord_excel, lat_coord_excel = (np.nan, np.nan)
            if sorted_flow_columns_excel and flat_idx < len(sorted_flow_columns_excel):
                col_name_excel = sorted_flow_columns_excel[flat_idx]
                if sensor_info_lookup_excel and col_name_excel in sensor_info_lookup_excel:
                    lon_coord_excel = sensor_info_lookup_excel[col_name_excel]['lon']
                    lat_coord_excel = sensor_info_lookup_excel[col_name_excel]['lat']

            row_data_excel = {
                '資料來源': model_key_excel,
                '網格座標_R': grid_r_coord_excel, '網格座標_C': grid_c_coord_excel, '經度': lon_coord_excel, '緯度': lat_coord_excel,
                'MSE': error_grids_eval_excel.get('MSE')[flat_idx] if error_grids_eval_excel.get('MSE') is not None and flat_idx < len(error_grids_eval_excel.get('MSE')) else np.nan,
                'MAE': error_grids_eval_excel.get('MAE')[flat_idx] if error_grids_eval_excel.get('MAE') is not None and flat_idx < len(error_grids_eval_excel.get('MAE')) else np.nan,
                'MAPE (AvgGrid)': error_grids_eval_excel.get('MAPE')[flat_idx] if error_grids_eval_excel.get('MAPE') is not None and flat_idx < len(error_grids_eval_excel.get('MAPE')) else np.nan,
                'SMAPE (AvgGrid)': error_grids_eval_excel.get('SMAPE')[flat_idx] if error_grids_eval_excel.get('SMAPE') is not None and flat_idx < len(error_grids_eval_excel.get('SMAPE')) else np.nan,
                'STDE_AvgGrid': error_grids_eval_excel.get('STDE_AvgGrid')[flat_idx] if error_grids_eval_excel.get('STDE_AvgGrid') is not None and flat_idx < len(error_grids_eval_excel.get('STDE_AvgGrid')) else np.nan,
                'MAPE (Overall)': 'N/A', 'SMAPE (Overall)': 'N/A', 'FID': 'N/A'
            }
            excel_rows_to_export.append(row_data_excel)

        avg_row_excel = {
            '資料來源': model_key_excel, '網格座標_R': '整體平均', '網格座標_C': f'({eval_type_log})', '經度': '', '緯度': '',
            'MSE': metrics_eval_excel.get('mse', np.nan), 'MAE': metrics_eval_excel.get('mae', np.nan),
            'MAPE (AvgGrid)': metrics_eval_excel.get('mape_avg_grid', np.nan), 
            'SMAPE (AvgGrid)': metrics_eval_excel.get('smape_avg_grid', np.nan),
            'MAPE (Overall)': metrics_eval_excel.get('mape_overall', np.nan),
            'SMAPE (Overall)': metrics_eval_excel.get('smape_overall', np.nan),
            'FID': metrics_eval_excel.get('fid', np.nan),
            'STDE_AvgGrid': metrics_eval_excel.get('stde_avg_grid'),
            'STDE_Overall': metrics_eval_excel.get('stde_overall')
        }
        excel_rows_to_export.append(avg_row_excel)

    if previous_stage_model_eval_instance and \
    f"{current_stage_name.lower()}_model" in results and \
    f"{ConditionMode(current_stage_mode.value - 1).name.lower()}_model_on_{current_stage_name.lower()}_data" in results:

        cs_excel_key = f"{current_stage_name.lower()}_model"
        ps_excel_key = f"{ConditionMode(current_stage_mode.value - 1).name.lower()}_model_on_{current_stage_name.lower()}_data"
        prev_stage_log_name_excel = ConditionMode(current_stage_mode.value - 1).name if current_stage_mode.value > 1 else "Basemodel"
        diff_source_label_excel = f"Difference ({current_stage_name} - {prev_stage_log_name_excel})"
        excel_rows_to_export.append({'資料來源': f"--- {diff_source_label_excel} ({prefix}) ---"})

        metrics_cs_excel = results[cs_excel_key]
        metrics_ps_excel = results[ps_excel_key]
        error_grids_cs_excel = error_grids_all_models[cs_excel_key]
        error_grids_ps_excel = error_grids_all_models[ps_excel_key]

        indices_to_loop = grid_mask_flat_indices if grid_mask_flat_indices is not None else range(num_grid_cells_eval)

        for flat_idx in indices_to_loop:
            grid_r_coord_excel, grid_c_coord_excel = ('N/A', 'N/A') if not grid_idx_to_rc_map_excel else grid_idx_to_rc_map_excel.get(flat_idx, ('N/A','N/A'))
            lon_coord_excel, lat_coord_excel = (np.nan, np.nan)
            if sorted_flow_columns_excel and flat_idx < len(sorted_flow_columns_excel):
                col_name_excel = sorted_flow_columns_excel[flat_idx]
                if sensor_info_lookup_excel and col_name_excel in sensor_info_lookup_excel:
                    lon_coord_excel = sensor_info_lookup_excel[col_name_excel]['lon']
                    lat_coord_excel = sensor_info_lookup_excel[col_name_excel]['lat']

            diff_row_data_excel = {'資料來源': diff_source_label_excel, '網格座標_R': grid_r_coord_excel, '網格座標_C': grid_c_coord_excel, '經度': lon_coord_excel, '緯度': lat_coord_excel}
            for metric_key_upper_excel in ['MSE', 'MAE', 'MAPE', 'SMAPE', 'STDE_AvgGrid']:
                val_cs_excel = error_grids_cs_excel.get(metric_key_upper_excel)[flat_idx] if error_grids_cs_excel.get(metric_key_upper_excel) is not None and flat_idx < len(error_grids_cs_excel.get(metric_key_upper_excel)) else np.nan
                val_ps_excel = error_grids_ps_excel.get(metric_key_upper_excel)[flat_idx] if error_grids_ps_excel.get(metric_key_upper_excel) is not None and flat_idx < len(error_grids_ps_excel.get(metric_key_upper_excel)) else np.nan

                excel_metric_key_diff_grid = metric_key_upper_excel
                if metric_key_upper_excel == "MAPE": excel_metric_key_diff_grid = "MAPE (AvgGrid)"
                if metric_key_upper_excel == "SMAPE": excel_metric_key_diff_grid = "SMAPE (AvgGrid)"
                diff_row_data_excel[excel_metric_key_diff_grid] = val_cs_excel - val_ps_excel if not (np.isnan(val_cs_excel) or np.isnan(val_ps_excel)) else np.nan

            excel_rows_to_export.append(diff_row_data_excel)

        diff_avg_row_excel = {
            '資料來源': diff_source_label_excel, '網格座標_R': '整體平均差異', '網格座標_C': f'({eval_type_log})', '經度': '', '緯度': ''}
        for metric_key_lower_base_excel in ['mse', 'mae', 'mape_avg_grid', 'smape_avg_grid', 'mape_overall', 'smape_overall', 'fid', 'stde_avg_grid', 'stde_overall']:
            val_cs_avg_excel = metrics_cs_excel.get(metric_key_lower_base_excel, np.nan)
            val_ps_avg_excel = metrics_ps_excel.get(metric_key_lower_base_excel, np.nan)
            excel_col_name_diff = metric_key_lower_base_excel.upper().replace("AVG_GRID", "(AvgGrid)").replace("OVERALL", "(Overall)")
            diff_avg_row_excel[excel_col_name_diff] = val_cs_avg_excel - val_ps_avg_excel if not (np.isnan(val_cs_avg_excel) or np.isnan(val_ps_avg_excel)) else np.nan
        excel_rows_to_export.append(diff_avg_row_excel)

    if excel_rows_to_export:
        df_excel_export = pd.DataFrame(excel_rows_to_export)
        excel_column_order_final = ['資料來源', '網格座標_R', '網格座標_C', '經度', '緯度', 
                                'MSE', 'MAE', 'MAPE (AvgGrid)', 'SMAPE (AvgGrid)', 'STDE_AvgGrid',
                                'MAPE (Overall)', 'SMAPE (Overall)', 'STDE_Overall', 'FID']
        for col in excel_column_order_final:
            if col not in df_excel_export.columns: df_excel_export[col] = np.nan
        df_excel_export = df_excel_export.reindex(columns=excel_column_order_final)

        current_stage_save_dir_excel = config.get(f"{current_stage_name.lower()}_model_save_dir", config.get("save_dir_stage3", config.get("save_dir")))
        excel_final_path = os.path.join(current_stage_save_dir_excel, f"{prefix}_metrics_detailed.xlsx")
        try:
            df_excel_export.to_excel(excel_final_path, index=False, sheet_name=f"{current_stage_name}_Details")
            logger.info(f"{current_stage_name} 評估 ({prefix}, {eval_type_log}): 詳細評估指標已匯出至: {excel_final_path}")
        except Exception as e:
            logger.error(f"{current_stage_name} 評估 ({prefix}): 匯出 Excel 失敗: {e}")

    cs_pred_t_to_return = torch.cat(all_cs_generated_denorm_list, dim=0) if all_cs_generated_denorm_list else torch.empty(0)
    ps_pred_t_to_return = torch.cat(all_ps_generated_denorm_on_cs_data_list, dim=0) if all_ps_generated_denorm_on_cs_data_list else torch.empty(0)
    target_t_to_return = cs_target_all_t # 這個張量在函式前面已經是合併好的了

    # 【新增】將兩個模型的預測張量打包在一個字典裡，方便後續使用
    all_predictions = {}
    if cs_pred_t_to_return.numel() > 0:
        all_predictions[f"{current_stage_name.lower()}_model"] = cs_pred_t_to_return
    
    if ps_pred_t_to_return.numel() > 0:
        prev_stage_model_key = f"{ConditionMode(current_stage_mode.value -1).name.lower()}_model_on_{current_stage_name.lower()}_data"
        all_predictions[prev_stage_model_key] = ps_pred_t_to_return
    
    # 【修改】回傳包含新張量的新元組
    return results, error_grids_all_models, all_predictions, target_t_to_return

@torch.no_grad()
def evaluate_baseline_model_for_comparison(
    model_trained: 'DDPM3D', dataloader: DataLoader, inception_model_fid: nn.Module,
    config: Dict[str, Any], target_norm_stats: Dict[str, float],
    target_grid_stds: np.ndarray,
    target_overall_std: float,
    prefix: str = "eval_baseline",
    grid_mask_hw: Optional[np.ndarray] = None, 
    grid_mask_flat_indices: Optional[np.ndarray] = None 
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    
    model_trained.eval()
    target_mean, target_std = target_norm_stats['mean'], target_norm_stats['std']
    if target_std < 1e-6: target_std = 1.0 # 避免除以零

    all_generated_denorm, all_target_denorm = [], []
    all_generated_norm, all_target_norm = [], []

    pbar_baseline = tqdm(dataloader, desc=f"Baseline Eval ({prefix})", leave=False)
    for target_norm_b, cond_norm_b in pbar_baseline:
        target_norm, cond_norm = target_norm_b.to(config["device"]), cond_norm_b.to(config["device"])
        
        generated_norm = model_trained.sample(
             batch_size=target_norm.shape[0],
             mode=ConditionMode.BASELINE_EVAL,
             condition_args={"direct_condition": cond_norm}
        )
        
        generated_denorm = generated_norm * target_std + target_mean
        target_denorm = target_norm * target_std + target_mean
        all_generated_denorm.append(generated_denorm.cpu())
        all_target_denorm.append(target_denorm.cpu())
        all_generated_norm.append(generated_norm.cpu())
        all_target_norm.append(target_norm.cpu())

    if not all_target_denorm:
        logger.warning(f"Baseline 評估 ({prefix}): 無數據處理。")
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape_avg_grid": float('nan'), "smape_avg_grid": float('nan'), "mape_overall": float('nan'), "smape_overall": float('nan'), "fid": float('nan')}
        nan_grids_dict = {m: np.array([np.nan]) for m in ['MSE', 'MAE', 'MAPE', 'SMAPE', 'STDE_AvgGrid']}
        return nan_metrics, nan_grids_dict, torch.empty(0), torch.empty(0)


    pred_t = torch.cat(all_generated_denorm, dim=0)
    target_t = torch.cat(all_target_denorm, dim=0)
    epsilon = 1e-8

    mask_tensor = None
    if grid_mask_hw is not None:
        mask_tensor = torch.from_numpy(grid_mask_hw.astype(bool)).to(target_t.device)
        mask_tensor = mask_tensor.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(target_t)

    pred_for_metric = torch.masked_select(pred_t, mask_tensor) if mask_tensor is not None else pred_t.flatten()
    target_for_metric = torch.masked_select(target_t, mask_tensor) if mask_tensor is not None else target_t.flatten()

    mse = F.mse_loss(pred_for_metric, target_for_metric).item()
    mae = F.l1_loss(pred_for_metric, target_for_metric).item()

    actual_vals = torch.abs(target_for_metric); errors = torch.abs(target_for_metric - pred_for_metric)
    valid_mape_mask = actual_vals > config.get("mape_threshold", 1.0)
    mape_avg_grid = torch.mean((errors[valid_mape_mask] / actual_vals[valid_mape_mask]) * 100).item() if torch.any(valid_mape_mask) else float('inf')
    smape_denom = (actual_vals + torch.abs(pred_for_metric)) / 2.0 + epsilon
    smape_avg_grid = torch.mean((errors / smape_denom) * 100).item()
    mape_overall = (torch.sum(errors) / (torch.sum(actual_vals) + epsilon)).item() * 100
    smape_overall = (200.0 * torch.sum(errors) / (torch.sum(actual_vals + torch.abs(pred_for_metric)) + epsilon)).item()

    grid_stds_t = torch.from_numpy(target_grid_stds.reshape(1, 1, 1, config["H"], config["W"])).to(pred_t.device)
    standardized_error_t = torch.abs(pred_t - target_t) / (grid_stds_t + epsilon)
    stde_g_map = torch.mean(standardized_error_t, dim=0).squeeze().cpu().numpy()
    stde_g = stde_g_map.flatten()
    if grid_mask_flat_indices is not None:
        stde_avg_grid = np.nanmean(stde_g[grid_mask_flat_indices])
        target_overall_std_filtered = torch.std(target_for_metric).item()
        stde_overall = mae / (target_overall_std_filtered + epsilon if target_overall_std_filtered > epsilon else 1.0)
    else:
        stde_avg_grid = np.nanmean(stde_g)
        stde_overall = mae / (target_overall_std + epsilon if target_overall_std > epsilon else 1.0)

    # FID
    gen_fid_tensor = torch.cat(all_generated_norm, dim=0); real_fid_tensor = torch.cat(all_target_norm, dim=0)
    act_gen = get_activations(gen_fid_tensor, inception_model_fid, config["device"], config["fid_batch_size"])
    act_real = get_activations(real_fid_tensor, inception_model_fid, config["device"], config["fid_batch_size"])
    fid = calculate_fid(act_real, act_gen)
    
    results = {
        "mse": mse, "mae": mae, "mape_avg_grid": mape_avg_grid, 
        "smape_avg_grid": smape_avg_grid, "mape_overall": mape_overall, 
        "smape_overall": smape_overall, 
        "stde_avg_grid": stde_avg_grid, "stde_overall": stde_overall, 
        "fid": fid
    }
    
    pred_s, target_s = pred_t.squeeze(1).squeeze(1), target_t.squeeze(1).squeeze(1)
    mse_g = torch.mean((pred_s - target_s)**2, dim=0).cpu().numpy()
    mae_g = torch.mean(torch.abs(pred_s - target_s), dim=0).cpu().numpy()
    mape_g = torch.mean(torch.abs((target_s - pred_s) / (target_s + epsilon)) * 100, dim=0).cpu().numpy()
    smape_g = torch.mean((torch.abs(pred_s-target_s) / ((torch.abs(target_s)+torch.abs(pred_s))/2 + epsilon)) * 100, dim=0).cpu().numpy()
    
    error_grids = {
        'MSE': mse_g.flatten(), 'MAE': mae_g.flatten(), 
        'MAPE': mape_g.flatten(), 'SMAPE': smape_g.flatten(),
        'STDE_AvgGrid': stde_g
    }
    logger.info(f"Generating 6-map heatmap visualizations for Baseline model ({prefix})...")
    if pred_t.shape[0] > 0 and target_t.shape[0] > 0:
        # 繪製單一樣本 (sample 0)
        visualize_predictions_long_term(
            pred_t[0:1].clone().cpu(),
            target_t[0:1].clone().cpu(),
            config,
            sample_idx_to_plot=0,
            prefix=f"{prefix}_vs_STAGE4Target_sample0",
            grid_mask_hw=grid_mask_hw
        )
        # 繪製所有樣本的平均
        visualize_predictions_long_term(
            torch.mean(pred_t, dim=0, keepdim=True).clone().cpu(),
            torch.mean(target_t, dim=0, keepdim=True).clone().cpu(),
            config,
            sample_idx_to_plot=None,
            prefix=f"{prefix}_vs_STAGE4Target_avg",
            grid_mask_hw=grid_mask_hw
        )
    logger.info(f"Generating visualizations for Baseline model evaluation ({prefix})...")
    plot_grid_with_error_long_term(
        dataset_for_coords=dataloader.dataset,
        error_metrics_grids=error_grids,
        config=config,
        prefix=prefix, # prefix 此時應為 "final_eval_full_map_Baseline"
        grid_mask_flat_indices=grid_mask_flat_indices
    )
    
    return results, error_grids, pred_t, target_t


#%%
if __name__ == '__main__':
    logger.info(f"===== DDPM Multi-Stage (Up to Stage4) - Training and Evaluation =====")
    filtered_grid_mask_hw = None
    filtered_grid_indices_flat = None
    expanded_grid_mask_hw = None
    expanded_grid_indices_flat = None
    if CONFIG["coordinate_filter"].get("enabled", False):
        logger.info("Coordinate filter is ENABLED. Loading filter file...")
        filter_path = CONFIG["coordinate_filter"]["file_path"]
        r_col, c_col = CONFIG["coordinate_filter"]["r_col"], CONFIG["coordinate_filter"]["c_col"]
        
        if not os.path.exists(filter_path):
            logger.error(f"Coordinate filter file not found: {filter_path}. Disabling filter.")
        else:
            try:
                filter_df = pd.read_excel(filter_path)
                if r_col not in filter_df.columns or c_col not in filter_df.columns:
                    raise ValueError(f"Filter file '{filter_path}' is missing required columns: '{r_col}' or '{c_col}'.")
                
                H, W = CONFIG["H"], CONFIG["W"]
                filtered_grid_mask_hw = np.zeros((H, W), dtype=bool)
                
                filtered_coords = set(zip(filter_df[r_col], filter_df[c_col]))
                for r, c in filtered_coords:
                    if 0 <= r < H and 0 <= c < W:
                        filtered_grid_mask_hw[r, c] = True
                
                filtered_grid_indices_flat = np.where(filtered_grid_mask_hw.flatten())[0]
                
                if len(filtered_grid_indices_flat) == 0:
                    logger.warning("Coordinate filter is enabled, but no valid coordinates were loaded. Disabling filter.")
                    filtered_grid_mask_hw = None
                    filtered_grid_indices_flat = None
                else:
                    logger.info(f"Loaded {len(filtered_grid_indices_flat)} coordinates to filter for evaluation.")

                    if filtered_grid_mask_hw is not None:
                        logger.info("正在根據原始篩選點，建立擴散後的 3x3 網格遮罩...")
                        H, W = CONFIG["H"], CONFIG["W"]
                        # 先複製原始遮罩，確保原始點也被包含
                        expanded_grid_mask_hw = filtered_grid_mask_hw.copy()
                        
                        # 找出原始遮罩中所有 True 的點的座標
                        original_rows, original_cols = np.where(filtered_grid_mask_hw)

                        # 遍歷每一個原始點
                        for r, c in zip(original_rows, original_cols):
                            # 擴散到 3x3 的鄰域 (包含中心點)
                            for dr in range(-1, 2):  # dr 會是 -1, 0, 1
                                for dc in range(-1, 2):  # dc 會是 -1, 0, 1
                                    nr, nc = r + dr, c + dc
                                    
                                    # 確保鄰近點的座標沒有超出網格邊界
                                    if 0 <= nr < H and 0 <= nc < W:
                                        expanded_grid_mask_hw[nr, nc] = True
                        
                        # 計算擴散後遮罩的扁平化索引
                        expanded_grid_indices_flat = np.where(expanded_grid_mask_hw.flatten())[0]
                        
                        logger.info(f"原始遮罩包含 {len(original_rows)} 個網格點。")
                        logger.info(f"擴散後的遮罩包含 {len(expanded_grid_indices_flat)} 個網格點。")

            except Exception as e:
                logger.error(f"Failed to load coordinate filter, disabling feature. Error: {e}")
                filtered_grid_mask_hw = None
                filtered_grid_indices_flat = None
    else:
        logger.info("Coordinate filter is DISABLED.")

    config_for_log = CONFIG.copy()
    keys_to_remove_from_log = [
        "cached_basemodel_sorted_flow_columns",
        "cached_basemodel_selected_sensor_info",
        "cached_basemodel_grid_idx_to_rc_map"
    ]
    for key in keys_to_remove_from_log:
        if key in config_for_log:
            # 可以選擇完全移除，或者賦予一個簡短的佔位符
            # config_for_log[key] = "<data omitted>" 
            del config_for_log[key] 

    logger.info(f"Full CONFIG (selected fields): {json.dumps(config_for_log, indent=2, ensure_ascii=False)}")

    # --- 載入完整數據 ---
    full_df = pd.read_csv(CONFIG["data_path"])
    if 'hoilday' in full_df.columns and 'holiday' not in full_df.columns:
        logger.info("偵測到欄位名稱 'hoilday'，自動更名為 'holiday' 以修正拼寫錯誤。")
        full_df.rename(columns={'hoilday': 'holiday'}, inplace=True)
    logger.info(f"已載入資料: {CONFIG['data_path']}. 形狀: {full_df.shape}")
    # === 新增步驟：創建「組合特徵」欄位 ===
    if '月' in full_df.columns and '日' in full_df.columns and '年' in full_df.columns:
        full_df['date_combined'] = (full_df['年'] - 2018) * 365 + full_df['月'] * 31 + full_df['日']
        logger.info("已成功創建 'month_day_cdate_combinedombined' 組合特徵欄位。")
    else:
        raise ValueError("DataFrame 中缺少 '月' 或 '日' 欄位，無法創建組合特徵。")

    # === 步驟 1: 載入預訓練的 Basemodel (用於生成後續階段的條件) ===
    BASEMODEL_CHECKPOINT_PATH = CONFIG["basemodel_checkpoint"]
    if not os.path.exists(BASEMODEL_CHECKPOINT_PATH):
        # 如果 Basemodel 必須存在，則應該拋出錯誤
        logger.error(f"CRITICAL: 未找到 Basemodel 檢查點: {BASEMODEL_CHECKPOINT_PATH}。程式可能無法繼續。")
        raise FileNotFoundError(f"未找到 Basemodel 檢查點: {BASEMODEL_CHECKPOINT_PATH}")
        
    logger.info(f"===== 載入 Basemodel (for output generation) 從: {BASEMODEL_CHECKPOINT_PATH} =====")
    chkpt_basemodel = torch.load(BASEMODEL_CHECKPOINT_PATH, map_location=CONFIG["device"], weights_only=False)
    if 'ddpm_state_dict' not in chkpt_basemodel:
        raise KeyError(f"Basemodel 檢查點 {BASEMODEL_CHECKPOINT_PATH} 中未找到 'ddpm_state_dict'。")
    
    config_basemodel_original = chkpt_basemodel.get('config', CONFIG) 

    CONFIG["cached_basemodel_selected_sensor_info"] = chkpt_basemodel.get('selected_sensor_info')
    CONFIG["cached_basemodel_grid_idx_to_rc_map"] = chkpt_basemodel.get('grid_idx_to_rc_map')
    CONFIG["cached_basemodel_sorted_flow_columns"] = chkpt_basemodel.get('sorted_flow_columns')
    
    if 'norm_stats_flow' not in chkpt_basemodel: # sorted_flow_columns 已在上面 get
        raise ValueError("Basemodel 檢查點必須包含 'norm_stats_flow'。")
    basemodel_norm_stats_source = chkpt_basemodel['norm_stats_flow']
    CONFIG["cached_basemodel_mean"] = float(basemodel_norm_stats_source['mean'])
    CONFIG["cached_basemodel_std"] = float(basemodel_norm_stats_source['std'])
    if CONFIG["cached_basemodel_std"] < 1e-6: CONFIG["cached_basemodel_std"] = 1.0

    if not all(CONFIG.get(k) for k in ["cached_basemodel_selected_sensor_info", 
                                        "cached_basemodel_grid_idx_to_rc_map", 
                                        "cached_basemodel_sorted_flow_columns"]):
        raise ValueError("Basemodel 檢查點缺少必要的網格映射資訊。無法繼續。")
    else:
        logger.info("成功從 Basemodel 檢查點加載網格映射資訊到 CONFIG。")
        logger.info(f"cached_basemodel_mean = {CONFIG['cached_basemodel_mean']:.4f}")
        logger.info(f"cached_basemodel_std = {CONFIG['cached_basemodel_std']:.4f}")

    basemodel_unet = UNet3D(
        config_basemodel_original.get("image_channels", CONFIG["image_channels"]),
        config_basemodel_original.get("base_channels_unet", CONFIG["base_channels_unet"]),
        config_basemodel_original.get("time_emb_dim", CONFIG["time_emb_dim"]),
        config_basemodel_original.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        dropout_rate=config_basemodel_original.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
    ).to(CONFIG["device"])

    basemodel_for_output_generation = DDPM3D(
        unet_model=basemodel_unet,
        timesteps=config_basemodel_original.get("timesteps", CONFIG["timesteps"]),
        image_size=(config_basemodel_original.get("D", CONFIG["D"]),
                     config_basemodel_original.get("H", CONFIG["H"]),
                     config_basemodel_original.get("W", CONFIG["W"])),
        image_channels=config_basemodel_original.get("image_channels", CONFIG["image_channels"]),
        condition_input_channels=config_basemodel_original.get("condition_input_channels", 2),
        condition_encode_dim=config_basemodel_original.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        beta_start=config_basemodel_original.get("beta_start", CONFIG["beta_start"]),
        beta_end=config_basemodel_original.get("beta_end", CONFIG["beta_end"]),
        device=CONFIG["device"]  
    )
    basemodel_for_output_generation.load_state_dict(chkpt_basemodel['ddpm_state_dict'])
    basemodel_for_output_generation.eval()
    logger.info(f"Basemodel (for output generation) 載入完成。")
    logger.info("正在從【整個原始資料集】計算全域目標流量標準差 (用於 STDE 指標)...")
    
    # 確保 'sorted_flow_columns' 已經從 Basemodel 檢查點載入
    if not CONFIG["cached_basemodel_sorted_flow_columns"]:
        raise ValueError("未能從 Basemodel 檢查點載入 'sorted_flow_columns'，無法計算標準差。")
        
    all_flow_data = full_df[CONFIG["cached_basemodel_sorted_flow_columns"]].values.astype(np.float32)
    
    # 計算全域的、逐網格的標準差
    global_target_grid_stds = np.std(all_flow_data, axis=0)
    # 計算全域的、總體的標準差
    global_target_overall_std = np.std(all_flow_data)
    
    logger.info(f"  - 全域逐網格標準差 (Global Grid STDs) shape: {global_target_grid_stds.shape}")
    logger.info(f"  - 全域總體標準差 (Global Overall STD): {global_target_overall_std:.4f}")

    # === 步驟 2: 準備 Basemodel 輸出 (作為S2條件) ===
    df_for_bm_output_gen = full_df.copy() 
    basemodel_outputs_cache_filepath = os.path.join(CONFIG["cache_dir_full_path"], CONFIG["cached_basemodel_outputs_for_s2_filename"])
    all_bm_outputs_s2_np_cond_normalized = None
    pred_bs = CONFIG.get("eval_batch_size", 64)
    if os.path.exists(basemodel_outputs_cache_filepath):
        try:
            logger.info(f"正在從快取檔案載入 Basemodel 輸出: {basemodel_outputs_cache_filepath}")
            all_bm_outputs_s2_np_cond_normalized = np.load(basemodel_outputs_cache_filepath)
            if all_bm_outputs_s2_np_cond_normalized.shape[0] != len(df_for_bm_output_gen):
                logger.warning(f"快取的 Basemodel 輸出樣本數與預期不符，將重新生成。")
                all_bm_outputs_s2_np_cond_normalized = None
            else:
                logger.info(f"  快取載入的 Basemodel 輸出 (正規化) - MIN: {np.min(all_bm_outputs_s2_np_cond_normalized):.4f}, MAX: {np.max(all_bm_outputs_s2_np_cond_normalized):.4f}, MEAN: {np.mean(all_bm_outputs_s2_np_cond_normalized):.4f}, STD: {np.std(all_bm_outputs_s2_np_cond_normalized):.4f}")
        except Exception as e:
            logger.error(f"從快取檔案載入 Basemodel 輸出失敗: {e}。將重新生成。")
            all_bm_outputs_s2_np_cond_normalized = None
            
    if all_bm_outputs_s2_np_cond_normalized is None:
        logger.info(f"生成 Basemodel 輸出作為後續階段條件...")
        hours_for_bm_scalar = torch.tensor(df_for_bm_output_gen['時'].values.astype(int), dtype=torch.long)
        if 'holiday' not in df_for_bm_output_gen and 'hoilday' in df_for_bm_output_gen:
           df_for_bm_output_gen.rename(columns={"hoilday": "holiday"}, inplace=True)
        is_holiday_for_bm_scalar = torch.tensor(df_for_bm_output_gen['holiday'].astype(int).values, dtype=torch.long)
        
        bm_outputs_list_cond = []
        basemodel_for_output_generation.eval() # 確保是評估模式
        with torch.no_grad():
            for i in tqdm(range(0, len(df_for_bm_output_gen), pred_bs), desc="Generating Basemodel Outputs"):
                b_hrs = hours_for_bm_scalar[i:i+pred_bs].to(CONFIG["device"])
                b_hols = is_holiday_for_bm_scalar[i:i+pred_bs].to(CONFIG["device"])
                
                condition_args_bm = {"hour_scalars_batch": b_hrs, "is_holiday_scalars_batch": b_hols}
                bm_pred_norm_b = basemodel_for_output_generation.sample(len(b_hrs), ConditionMode.BASEMODEL, condition_args_bm)
                bm_pred_denorm_b = bm_pred_norm_b * CONFIG["cached_basemodel_std"] + CONFIG["cached_basemodel_mean"]
                bm_outputs_list_cond.append(bm_pred_denorm_b.cpu().numpy())
            
        all_bm_outputs_s2_np_cond = np.concatenate(bm_outputs_list_cond, axis=0)
        if all_bm_outputs_s2_np_cond.shape[1] != CONFIG["image_channels"]:
             logger.warning(f"Basemodel output for Stage2 conditions has {all_bm_outputs_s2_np_cond.shape[1]} channels, expected {CONFIG['image_channels']}. Using first {CONFIG['image_channels']} channel(s).")
             all_bm_outputs_s2_np_cond = all_bm_outputs_s2_np_cond[:, 0:CONFIG["image_channels"], ...]
        logger.info(f"Basemodel 輸出 (反正規化) 生成完畢, 形狀: {all_bm_outputs_s2_np_cond.shape}")
        
        all_bm_outputs_s2_np_cond_normalized = (all_bm_outputs_s2_np_cond - CONFIG["cached_basemodel_mean"]) / CONFIG["cached_basemodel_std"]
        try:
            np.save(basemodel_outputs_cache_filepath, all_bm_outputs_s2_np_cond_normalized)
            logger.info(f"Basemodel 輸出已儲存到快取: {basemodel_outputs_cache_filepath}")
        except Exception as e:
            logger.error(f"儲存 Basemodel 輸出到快取失敗: {e}")
        logger.info(f"  生成並正規化後的 Basemodel 輸出 (作為S2條件) - MIN: {np.min(all_bm_outputs_s2_np_cond_normalized):.4f}, MAX: {np.max(all_bm_outputs_s2_np_cond_normalized):.4f}, MEAN: {np.mean(all_bm_outputs_s2_np_cond_normalized):.4f}, STD: {np.std(all_bm_outputs_s2_np_cond_normalized):.4f}")

    # === 步驟 3: 載入預訓練的 Stage2 模型 ===
    logger.info(f"===== STAGE 2: 載入預訓練模型 ({CONFIG['stage2_model_name']}) =====")
    stage2_checkpoint_load_path = CONFIG["stage2_checkpoint_full_path"]
    if not os.path.exists(stage2_checkpoint_load_path):
        raise FileNotFoundError(f"未找到 Stage2 檢查點: {stage2_checkpoint_load_path}。此流程中 Stage2 模型不進行訓練，必須提供。")

    chkpt_s2_eval = torch.load(stage2_checkpoint_load_path, map_location=CONFIG["device"], weights_only=False)
    config_from_s2_chkpt = chkpt_s2_eval.get('config_snapshot_at_save', CONFIG)
    
    s2_unet_eval = UNet3D(
        config_from_s2_chkpt.get("image_channels", CONFIG["image_channels"]),
        config_from_s2_chkpt.get("base_channels_unet", CONFIG["base_channels_unet"]),
        config_from_s2_chkpt.get("time_emb_dim", CONFIG["time_emb_dim"]),
        config_from_s2_chkpt.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        dropout_rate=config_from_s2_chkpt.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
    ).to(CONFIG["device"])

    final_s2_model_to_eval = DDPM3D(
        unet_model=s2_unet_eval,
        timesteps=config_from_s2_chkpt.get("timesteps", CONFIG["timesteps"]),
        image_size=(config_from_s2_chkpt.get("D", CONFIG["D"]), config_from_s2_chkpt.get("H", CONFIG["H"]), config_from_s2_chkpt.get("W", CONFIG["W"])),
        image_channels=config_from_s2_chkpt.get("image_channels", CONFIG["image_channels"]),
        condition_input_channels=config_from_s2_chkpt.get("condition_input_channels", CONFIG.get("stage2_ddpm_condition_input_channels",2)),
        condition_encode_dim=config_from_s2_chkpt.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        beta_start=config_from_s2_chkpt.get("beta_start", CONFIG["beta_start"]),
        beta_end=config_from_s2_chkpt.get("beta_end", CONFIG["beta_end"]),
        device=CONFIG["device"]
    )
    final_s2_model_to_eval.load_state_dict(chkpt_s2_eval['ddpm_state_dict'])
    final_s2_model_to_eval.eval()
    logger.info(f"Stage2 模型 (Epoch {chkpt_s2_eval.get('epoch','未知')}) 載入完成。")
    
    s2_new_cond_stats_for_subsequent_stages = chkpt_s2_eval.get('new_cond_feature_norm_stats') 
    stage2_target_norm_stats_for_eval = chkpt_s2_eval.get('norm_stats_stage2_target')
    if s2_new_cond_stats_for_subsequent_stages is None or stage2_target_norm_stats_for_eval is None:
        raise ValueError("Stage2 檢查點缺少必要的正規化統計量 (new_cond_feature_norm_stats 或 norm_stats_stage2_target)。")

    # === 步驟 4: 準備 Stage2 輸出 (作為 Stage3 條件1) ===
    df_for_s2_output_gen = full_df.copy()
    stage2_outputs_cache_filepath = os.path.join(CONFIG["cache_dir_full_path"], CONFIG["cached_stage2_outputs_for_s3_filename"])
    all_s2_outputs_s3_np_cond_normalized = None

    if os.path.exists(stage2_outputs_cache_filepath):
        try:
            logger.info(f"正在從快取檔案載入 Stage2 ({CONFIG['stage2_model_name']}) 輸出: {stage2_outputs_cache_filepath}")
            all_s2_outputs_s3_np_cond_normalized = np.load(stage2_outputs_cache_filepath)
            if all_s2_outputs_s3_np_cond_normalized.shape[0] != len(df_for_s2_output_gen):
                logger.warning(f"快取的 Stage2 ({CONFIG['stage2_model_name']}) 輸出樣本數與預期不符 ({all_s2_outputs_s3_np_cond_normalized.shape[0]} vs {len(df_for_s2_output_gen)})，將重新生成。")
                all_s2_outputs_s3_np_cond_normalized = None
            else:
                logger.info(f"  快取載入的 Stage2 ({CONFIG['stage2_model_name']}) 輸出 (正規化) - MIN: {np.min(all_s2_outputs_s3_np_cond_normalized):.4f}, MAX: {np.max(all_s2_outputs_s3_np_cond_normalized):.4f}, MEAN: {np.mean(all_s2_outputs_s3_np_cond_normalized):.4f}, STD: {np.std(all_s2_outputs_s3_np_cond_normalized):.4f}")
        except Exception as e:
            logger.error(f"從快取檔案載入 Stage2 ({CONFIG['stage2_model_name']}) 輸出失敗: {e}。將重新生成。")
            all_s2_outputs_s3_np_cond_normalized = None
            
    if all_s2_outputs_s3_np_cond_normalized is None:
        logger.info(f"生成 Stage2 ({CONFIG['stage2_model_name']}) 模型輸出作為 Stage3 條件...")
        s2_cond2_orig_vals = pd.to_numeric(df_for_s2_output_gen[CONFIG["stage2_new_condition_feature_column"]], errors='coerce').values
        s2_cond2_m = s2_new_cond_stats_for_subsequent_stages['mean']
        s2_cond2_s = s2_new_cond_stats_for_subsequent_stages['std']
        if s2_cond2_s < 1e-6: s2_cond2_s=1.0
        s2_cond2_grids_list = []
        for v in s2_cond2_orig_vals:
            nv = (v - s2_cond2_m)/s2_cond2_s if not np.isnan(v) else 0.0
            s2_cond2_grids_list.append(np.full((1,1,CONFIG["D"],CONFIG["H"],CONFIG["W"]), nv, dtype=np.float32))
        all_s2_new_feat_grids_for_s2_sample = np.concatenate(s2_cond2_grids_list, axis=0)
        
        s2_outputs_list = []
        final_s2_model_to_eval.eval() # 確保是評估模式
        with torch.no_grad():
            for i in tqdm(range(0, len(df_for_s2_output_gen), pred_bs), desc=f"Generating S2 ({CONFIG['stage2_model_name']}) Outputs for S3 Cond"):
                bm_out_b = torch.from_numpy(all_bm_outputs_s2_np_cond_normalized[i:i+pred_bs]).to(CONFIG["device"])
                s2_new_feat_b = torch.from_numpy(all_s2_new_feat_grids_for_s2_sample[i:i+pred_bs]).to(CONFIG["device"])
                cond_args = {"basemodel_output_grid_batch":bm_out_b, "stage2_new_condition_feature_grid_batch":s2_new_feat_b}
                s2_pred_norm = final_s2_model_to_eval.sample(len(bm_out_b),ConditionMode.STAGE2, cond_args)
                s2_outputs_list.append(s2_pred_norm.cpu().numpy())
        all_s2_outputs_s3_np_cond_normalized = np.concatenate(s2_outputs_list, axis=0)
        if all_s2_outputs_s3_np_cond_normalized.shape[1]!=1: all_s2_outputs_s3_np_cond_normalized=all_s2_outputs_s3_np_cond_normalized[:,0:1,...]
        
        try:
            np.save(stage2_outputs_cache_filepath, all_s2_outputs_s3_np_cond_normalized)
            logger.info(f"Stage2 ({CONFIG['stage2_model_name']}) 輸出已儲存到快取: {stage2_outputs_cache_filepath}")
        except Exception as e:
            logger.error(f"儲存 Stage2 ({CONFIG['stage2_model_name']}) 輸出到快取失敗: {e}")
        logger.info(f"  生成並正規化後的 Stage2 ({CONFIG['stage2_model_name']}) 模型輸出 (作為S3條件) - MIN: {np.min(all_s2_outputs_s3_np_cond_normalized):.4f}, MAX: {np.max(all_s2_outputs_s3_np_cond_normalized):.4f}, MEAN: {np.mean(all_s2_outputs_s3_np_cond_normalized):.4f}, STD: {np.std(all_s2_outputs_s3_np_cond_normalized):.4f}")


    # === 步驟 5: 載入預訓練的 Stage3 模型 ===
    logger.info(f"===== STAGE 3: 載入預訓練模型 ({CONFIG['stage3_model_name']}) =====")
    stage3_checkpoint_load_path = CONFIG["stage3_checkpoint_full_path"] # S3的最佳模型路徑
    if not os.path.exists(stage3_checkpoint_load_path):
        raise FileNotFoundError(f"未找到 Stage3 檢查點: {stage3_checkpoint_load_path}。此流程中 Stage3 模型不進行訓練，必須提供。")

    chkpt_s3_eval = torch.load(stage3_checkpoint_load_path, map_location=CONFIG["device"], weights_only=False)
    config_from_s3_chkpt = chkpt_s3_eval.get('config_snapshot_at_save', CONFIG)
    s3_unet_eval = UNet3D( # ... (使用 config_from_s3_chkpt 或 CONFIG 初始化 UNet) ...
        config_from_s3_chkpt.get("image_channels", CONFIG["image_channels"]),
        config_from_s3_chkpt.get("base_channels_unet", CONFIG["base_channels_unet"]),
        config_from_s3_chkpt.get("time_emb_dim", CONFIG["time_emb_dim"]),
        config_from_s3_chkpt.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        dropout_rate=config_from_s3_chkpt.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
    ).to(CONFIG["device"])
    final_s3_model_to_eval = DDPM3D( # ... (使用 config_from_s3_chkpt 或 CONFIG 初始化 DDPM) ...
        unet_model=s3_unet_eval,
        timesteps=config_from_s3_chkpt.get("timesteps", CONFIG["timesteps"]),
        image_size=(config_from_s3_chkpt.get("D", CONFIG["D"]), config_from_s3_chkpt.get("H", CONFIG["H"]), config_from_s3_chkpt.get("W", CONFIG["W"])),
        image_channels=config_from_s3_chkpt.get("image_channels", CONFIG["image_channels"]),
        condition_input_channels=config_from_s3_chkpt.get("condition_input_channels", CONFIG.get("stage3_ddpm_condition_input_channels",2)),
        condition_encode_dim=config_from_s3_chkpt.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        beta_start=config_from_s3_chkpt.get("beta_start", CONFIG["beta_start"]),
        beta_end=config_from_s3_chkpt.get("beta_end", CONFIG["beta_end"]),
        device=CONFIG["device"]
    )
    final_s3_model_to_eval.load_state_dict(chkpt_s3_eval['ddpm_state_dict'])
    final_s3_model_to_eval.eval()
    logger.info(f"Stage3 模型 (Epoch {chkpt_s3_eval.get('epoch','未知')}) 載入完成。")

    s3_new_cond_stats_for_s4_dataset = chkpt_s3_eval.get('s3_new_cond_feature_norm_stats') # Stage3 新特徵的統計量
    stage3_target_norm_stats_for_eval = chkpt_s3_eval.get('norm_stats_stage3_target')    # Stage3 目標的統計量
    if s3_new_cond_stats_for_s4_dataset is None or stage3_target_norm_stats_for_eval is None:
        raise ValueError("Stage3 檢查點缺少必要的正規化統計量 (s3_new_cond_feature_norm_stats 或 norm_stats_stage3_target)。")


    # === 步驟 6: 準備 Stage3 輸出 (作為 Stage4 條件1) ===
    df_for_s3_output_gen = full_df.copy()
    stage3_outputs_cache_filepath = os.path.join(CONFIG["cache_dir_full_path"], CONFIG["cached_stage3_outputs_for_s4_filename"])
    all_s3_outputs_s4_np_cond_normalized = None
    if os.path.exists(stage3_outputs_cache_filepath):
        try:
            logger.info(f"正在從快取檔案載入 Stage3 ({CONFIG['stage3_model_name']}) 輸出: {stage3_outputs_cache_filepath}")
            all_s3_outputs_s4_np_cond_normalized = np.load(stage3_outputs_cache_filepath)
            if all_s3_outputs_s4_np_cond_normalized.shape[0] != len(df_for_s3_output_gen):
                logger.warning(f"快取的 Stage3 ({CONFIG['stage3_model_name']}) 輸出樣本數與預期不符 ({all_s3_outputs_s4_np_cond_normalized.shape[0]} vs {len(df_for_s3_output_gen)})，將重新生成。")
                all_s3_outputs_s4_np_cond_normalized = None
            else:
                logger.info(f"  快取載入的 Stage3 ({CONFIG['stage3_model_name']}) 輸出 (正規化) - MIN: {np.min(all_s3_outputs_s4_np_cond_normalized):.4f}, MAX: {np.max(all_s3_outputs_s4_np_cond_normalized):.4f}, MEAN: {np.mean(all_s3_outputs_s4_np_cond_normalized):.4f}, STD: {np.std(all_s3_outputs_s4_np_cond_normalized):.4f}")
        except Exception as e:
            logger.error(f"從快取檔案載入 Stage3 ({CONFIG['stage3_model_name']}) 輸出失敗: {e}。將重新生成。")
            all_s3_outputs_s4_np_cond_normalized = None

    if all_s3_outputs_s4_np_cond_normalized is None:
        logger.info(f"生成 Stage3 ({CONFIG['stage3_model_name']}) 模型輸出作為 Stage4 條件...")
        s3_cond1_for_gen = all_s2_outputs_s3_np_cond_normalized # S2 的輸出
        s3_cond2_orig_vals = pd.to_numeric(df_for_s3_output_gen[CONFIG["stage3_new_condition_feature_column"]], errors='coerce').values
        s3_cond2_m = s3_new_cond_stats_for_s4_dataset['mean']
        s3_cond2_s = s3_new_cond_stats_for_s4_dataset['std']
        if s3_cond2_s < 1e-6: s3_cond2_s = 1.0
        s3_cond2_grids_list = []
        for v in s3_cond2_orig_vals:
            nv = (v - s3_cond2_m)/s3_cond2_s if not np.isnan(v) else 0.0
            s3_cond2_grids_list.append(np.full((1,1,CONFIG["D"],CONFIG["H"],CONFIG["W"]), nv, dtype=np.float32))
        all_s3_new_feat_grids_for_s3_sample = np.concatenate(s3_cond2_grids_list, axis=0)
        
        s3_outputs_list = []
        final_s3_model_to_eval.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(df_for_s3_output_gen), pred_bs), desc=f"Generating S3 ({CONFIG['stage3_model_name']}) Outputs for S4 Cond"):
                s2_out_b = torch.from_numpy(s3_cond1_for_gen[i:i+pred_bs]).to(CONFIG["device"])
                s3_new_feat_b = torch.from_numpy(all_s3_new_feat_grids_for_s3_sample[i:i+pred_bs]).to(CONFIG["device"])
                cond_args = {"stage2_output_grid_batch_for_s3":s2_out_b, "stage3_new_condition_feature_grid_batch":s3_new_feat_b}
                s3_pred_norm = final_s3_model_to_eval.sample(len(s2_out_b), ConditionMode.STAGE3, cond_args)
                s3_outputs_list.append(s3_pred_norm.cpu().numpy())
        all_s3_outputs_s4_np_cond_normalized = np.concatenate(s3_outputs_list, axis=0)
        if all_s3_outputs_s4_np_cond_normalized.shape[1]!=1: all_s3_outputs_s4_np_cond_normalized=all_s3_outputs_s4_np_cond_normalized[:,0:1,...]
        try:
            np.save(stage3_outputs_cache_filepath, all_s3_outputs_s4_np_cond_normalized)
            logger.info(f"Stage3 ({CONFIG['stage3_model_name']}) 輸出已儲存到快取: {stage3_outputs_cache_filepath}")
        except Exception as e:
            logger.error(f"儲存 Stage3 ({CONFIG['stage3_model_name']}) 輸出到快取失敗: {e}")
        logger.info(f"  生成並正規化後的 Stage3 ({CONFIG['stage3_model_name']}) 模型輸出 (作為S4條件) - MIN: {np.min(all_s3_outputs_s4_np_cond_normalized):.4f}, MAX: {np.max(all_s3_outputs_s4_np_cond_normalized):.4f}, MEAN: {np.mean(all_s3_outputs_s4_np_cond_normalized):.4f}, STD: {np.std(all_s3_outputs_s4_np_cond_normalized):.4f}")

    # === 新增步驟 6.5: 根據 Stage4 複合條件過濾數據 ===
    s4_mode = CONFIG["stage4_config"]["mode"]
    logger.info(f"===== STAGE 4 Pre-processing: Filtering data in '{s4_mode}' mode =====")

    if s4_mode == 'event':
        event_params = CONFIG["stage4_config"]["event_params"]
        event_filter_config = event_params["event_filter"]
        
        event_file_path = event_filter_config["file_path"]
        year_col, month_col, day_col = event_filter_config["year_col"], event_filter_config["month_col"], event_filter_config["day_col"]
        
        if not os.path.exists(event_file_path):
            raise FileNotFoundError(f"找不到活動日期 Excel 檔案: {event_file_path}")
        
        logger.info(f"正在從 {event_file_path} 讀取活動日期...")
        events_df = pd.read_excel(event_file_path)

        required_cols = [year_col, month_col, day_col]
        if not all(col in events_df.columns for col in required_cols):
            raise ValueError(f"Excel 檔案 '{event_file_path}' 中缺少必要的欄位，需要: {required_cols}")

        event_date_set = set(zip(events_df[year_col], events_df[month_col], events_df[day_col]))
        logger.info(f"從檔案中提取了 {len(event_date_set)} 個不重複的活動日期 (年, 月, 日)。")

        final_mask = full_df.apply(lambda row: (row['年'], row['月'], row['日']) in event_date_set, axis=1)

    elif s4_mode == 'feature':
        feature_params = CONFIG["stage4_config"]["feature_params"]
        column = feature_params["new_condition_feature_column"]
        operator = feature_params["new_conditional_operator"]
        value = feature_params["new_conditional_value"]
        
        logger.info(f"正在根據特徵條件篩選數據: {column} {operator} {value}")
        final_mask = create_condition_mask(full_df, column, operator, value)

    else:
        raise ValueError(f"不支援的 Stage4 模式: '{s4_mode}'。請檢查 CONFIG 設定。")

    # 後續的篩選和日誌記錄邏輯保持不變
    df_for_s4_specialist = full_df[final_mask].copy()
    num_total = len(full_df)
    num_filtered = len(df_for_s4_specialist)

    if num_filtered == 0:
        raise ValueError(f"在 Stage4 '{s4_mode}' 模式下，沒有任何數據滿足篩選條件，無法繼續。請檢查您的 CONFIG 設定或數據。")

    logger.info(f"數據過濾完成。總共 {num_total} 筆資料，滿足條件的有 {num_filtered} 筆 ({num_filtered/num_total:.2%})。")

    # 過濾之前階段生成的 NumPy 輸出陣列
    bm_outputs_for_s4_processing = all_bm_outputs_s2_np_cond_normalized[final_mask]
    s2_outputs_for_s4_processing = all_s2_outputs_s3_np_cond_normalized[final_mask]
    s3_outputs_for_s4_processing = all_s3_outputs_s4_np_cond_normalized[final_mask]

#%%
    # === 步驟 7: Stage4 數據準備與模型訓練 ===
    s4_mode = CONFIG["stage4_config"]["mode"]
    s4_params = CONFIG["stage4_config"][f"{s4_mode}_params"]
    model_name_for_log = s4_params["model_name"]
    logger.info(f"===== STAGE 4: 數據準備與模型訓練 ({model_name_for_log}) =====")
    df_for_s4_processing = full_df.copy()

    s4_indices_all = np.arange(len(df_for_s4_specialist))
    np.random.shuffle(s4_indices_all)
    s4_train_len = int(CONFIG["train_split_ratio"] * len(s4_indices_all))
    s4_val_len = int(CONFIG["val_split_ratio"] * len(s4_indices_all))
    s4_train_indices = s4_indices_all[:s4_train_len]
    s4_val_indices = s4_indices_all[s4_train_len : s4_train_len + s4_val_len]
    s4_test_indices = s4_indices_all[s4_train_len + s4_val_len:]
    logger.info(f"過濾後的 Stage4 資料分割: 訓練集={len(s4_train_indices)}, 驗證集={len(s4_val_indices)}, 測試集={len(s4_test_indices)}")

    train_dataset_s4 = MultiStageDataset(
        df_for_processing=df_for_s4_specialist.iloc[s4_train_indices], # <--- 修改點
        config=CONFIG,
        original_sorted_flow_columns=CONFIG["cached_basemodel_sorted_flow_columns"],
        current_stage_mode=ConditionMode.STAGE4,
        mode='train',
        basemodel_outputs_np=bm_outputs_for_s4_processing[s4_train_indices], # <--- 修改點
        s2_model_outputs_np=s2_outputs_for_s4_processing[s4_train_indices], # <--- 修改點
        s3_model_outputs_np=s3_outputs_for_s4_processing[s4_train_indices], # <--- 修改點
        s2_new_cond_feature_norm_stats=s2_new_cond_stats_for_subsequent_stages,
        s3_new_cond_feature_norm_stats=s3_new_cond_stats_for_s4_dataset,
    )
    s4_batch_size = CONFIG.get("batch_size")
    train_loader_s4_final = DataLoader(train_dataset_s4, batch_size=s4_batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True if len(train_dataset_s4) >= s4_batch_size else False)
    logger.info(f"Stage4 訓練數據集創建，含 {len(train_dataset_s4)} 樣本。")

    val_loader_s4_final = None
    if len(s4_val_indices) > 0:
        val_dataset_s4 = MultiStageDataset(
            df_for_processing=df_for_s4_specialist.iloc[s4_val_indices],
            config=CONFIG,
            original_sorted_flow_columns=CONFIG["cached_basemodel_sorted_flow_columns"],
            current_stage_mode=ConditionMode.STAGE4,
            mode='val',
            basemodel_outputs_np=bm_outputs_for_s4_processing[s4_val_indices], # 使用過濾後的陣列
            s2_model_outputs_np=s2_outputs_for_s4_processing[s4_val_indices], # 使用過濾後的陣列
            s3_model_outputs_np=s3_outputs_for_s4_processing[s4_val_indices], # 使用過濾後的陣列
            s2_new_cond_feature_norm_stats=s2_new_cond_stats_for_subsequent_stages,
            s3_new_cond_feature_norm_stats=s3_new_cond_stats_for_s4_dataset,
            s4_new_cond_feature_norm_stats=train_dataset_s4.norm_stats_s4_new_cond_feature,
            current_stage_avg_flow_map_dict_from_train=train_dataset_s4.average_flow_map_dict_current_stage,
            current_stage_target_norm_stats_from_train=train_dataset_s4.norm_stats_current_stage_target
        )
        
        val_loader_s4_final = DataLoader(
            val_dataset_s4, 
            batch_size=CONFIG["eval_batch_size"], 
            shuffle=False, 
            num_workers=CONFIG["num_workers"], 
            pin_memory=True
        )
        logger.info(f"Stage4 驗證數據集創建完成，含 {len(val_dataset_s4)} 筆樣本。")

    test_loader_s4_final = None
    if len(s4_test_indices) > 0:
        test_dataset_s4 = MultiStageDataset(
            df_for_processing=df_for_s4_specialist.iloc[s4_test_indices], # 使用過濾後的 df
            config=CONFIG,
            original_sorted_flow_columns=CONFIG["cached_basemodel_sorted_flow_columns"],
            current_stage_mode=ConditionMode.STAGE4,
            mode='test',
            # --- 主要修改點：使用步驟 6.5 中過濾後的 NumPy 陣列 ---
            basemodel_outputs_np=bm_outputs_for_s4_processing[s4_test_indices], # 使用過濾後的陣列
            s2_model_outputs_np=s2_outputs_for_s4_processing[s4_test_indices], # 使用過濾後的陣列
            s3_model_outputs_np=s3_outputs_for_s4_processing[s4_test_indices], # 使用過濾後的陣列

            s2_new_cond_feature_norm_stats=s2_new_cond_stats_for_subsequent_stages,
            s3_new_cond_feature_norm_stats=s3_new_cond_stats_for_s4_dataset,
            s4_new_cond_feature_norm_stats=train_dataset_s4.norm_stats_s4_new_cond_feature,
            current_stage_avg_flow_map_dict_from_train=train_dataset_s4.average_flow_map_dict_current_stage,
            current_stage_target_norm_stats_from_train=train_dataset_s4.norm_stats_current_stage_target
        )
        test_loader_s4_final = DataLoader(test_dataset_s4, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        logger.info(f"Stage4 測試數據集創建，含 {len(test_dataset_s4)} 樣本。")
#%%
    # --- Stage4 模型訓練迴圈 ---
    stage4_model: Optional[DDPM3D] = None 
    stage4_model_save_checkpoint_path_full = CONFIG["stage4_checkpoint_full_path"]
    
    if train_loader_s4_final:
        logger.info(f"===== STAGE 4: 模型訓練 ({model_name_for_log}) =====")
        # 實例化 Stage4 模型 (從S3初始化或從S4檢查點恢復)
        if os.path.exists(stage4_model_save_checkpoint_path_full) and CONFIG.get("resume_stage4_training", True): # 假設有 resume_stage4_training
            logger.info(f"準備從 Stage4 檢查點載入模型結構和權重: {stage4_model_save_checkpoint_path_full}")
            chkpt_s4_resume = torch.load(stage4_model_save_checkpoint_path_full, map_location=CONFIG["device"], weights_only=False)
            config_from_s4_chkpt_resume = chkpt_s4_resume.get('config_snapshot_at_save', CONFIG)
            s4_unet_resume = UNet3D(
                 config_from_s4_chkpt_resume.get("image_channels", CONFIG["image_channels"]),
                 config_from_s4_chkpt_resume.get("base_channels_unet", CONFIG["base_channels_unet"]),
                 config_from_s4_chkpt_resume.get("time_emb_dim", CONFIG["time_emb_dim"]),
                 config_from_s4_chkpt_resume.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
                 dropout_rate=config_from_s4_chkpt_resume.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
            ).to(CONFIG["device"])
            stage4_model = DDPM3D(
                unet_model=s4_unet_resume, 
                timesteps=config_from_s4_chkpt_resume.get("timesteps", CONFIG["timesteps"]),
                image_size=(config_from_s4_chkpt_resume.get("D", CONFIG["D"]), config_from_s4_chkpt_resume.get("H", CONFIG["H"]), config_from_s4_chkpt_resume.get("W", CONFIG["W"])),
                image_channels=config_from_s4_chkpt_resume.get("image_channels", CONFIG["image_channels"]),
                condition_input_channels=config_from_s4_chkpt_resume.get("condition_input_channels", CONFIG.get("stage4_ddpm_condition_input_channels",2)),
                condition_encode_dim=config_from_s4_chkpt_resume.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
                device=CONFIG["device"]
            )
            logger.info(f"Stage4 模型骨架已根據檢查點配置創建。")
        else:
            logger.info(f"將從 Stage3 模型 ({CONFIG['stage3_checkpoint_full_path']}) 初始化 Stage4 模型。")
            stage4_model = create_next_stage_model_from_previous_checkpoint(
                config_for_current_stage_and_global=CONFIG, # 傳入全局 CONFIG
                device=CONFIG["device"],
                current_stage_mode=ConditionMode.STAGE4 
            )
        
        if stage4_model:
            learning_rate_s4 = CONFIG.get("lr")
            optimizer_s4 = optim.AdamW(list(stage4_model.parameters()),
                               lr=learning_rate_s4,
                               weight_decay=CONFIG.get("weight_decay")  # 將 weight_decay 移到括號內
                              ) 
            scheduler_s4 = ReduceLROnPlateau(optimizer_s4, mode='min', 
                                             factor=CONFIG.get("lr_scheduler_factor"),
                                             patience=CONFIG.get("lr_scheduler_patience"),
                                             min_lr=CONFIG.get("lr_scheduler_min_lr"))
            start_epoch_s4 = 1
            best_val_loss_s4 = float('inf')
            early_stopping_counter_s4 = 0
            metrics_hist_s4 = {'train_loss':[], 'val_loss':[], 'lr':[]}
            epochs_to_run_s4 = CONFIG.get("epochs")

            if os.path.exists(stage4_model_save_checkpoint_path_full):
                logger.info(f"從 Stage4 檢查點恢復訓練狀態: {stage4_model_save_checkpoint_path_full}")
                # chkpt_s4_resume 已在上面載入 (如果模型是從 S4 檢查點初始化的)
                # 如果模型是從 S3 初始化的，但存在 S4 檢查點，則需要重新載入
                if not ('chkpt_s4_resume' in locals() and chkpt_s4_resume): 
                    chkpt_s4_resume = torch.load(stage4_model_save_checkpoint_path_full, map_location=CONFIG["device"], weights_only=False)

                stage4_model.load_state_dict(chkpt_s4_resume['ddpm_state_dict']) 
                optimizer_s4.load_state_dict(chkpt_s4_resume['optimizer_state_dict'])
                if 'scheduler_state_dict' in chkpt_s4_resume and chkpt_s4_resume['scheduler_state_dict']:
                     scheduler_s4.load_state_dict(chkpt_s4_resume['scheduler_state_dict'])
                start_epoch_s4 = chkpt_s4_resume.get('epoch', 0) + 1
                best_val_loss_s4 = chkpt_s4_resume.get('best_val_loss_s4', float('inf'))
                early_stopping_counter_s4 = chkpt_s4_resume.get('early_stopping_counter_s4', 0)
                metrics_hist_s4 = chkpt_s4_resume.get('metrics_hist_s4', {'train_loss':[], 'val_loss':[], 'lr':[]})
                logger.info(f"Stage4 訓練將從 epoch {start_epoch_s4} 開始。")

            logger.info(f"開始訓練 Stage4 模型: {model_name_for_log} for {epochs_to_run_s4} epochs...")

            epoch_pbar_s4 = tqdm(range(start_epoch_s4, epochs_to_run_s4 + 1), 
                                desc=f"Stage4 Training ({model_name_for_log})", 
                                leave=True, position=0, dynamic_ncols=True, unit="epoch")

            for epoch_s4_current in epoch_pbar_s4:
                # --- 訓練階段 ---
                stage4_model.train()
                total_train_loss_epoch_s4 = 0.0
                train_pbar_s4_loop = tqdm(train_loader_s4_final, 
                                        desc=f"Epoch {epoch_s4_current} [S4 Train]", 
                                        leave=False, position=1, dynamic_ncols=True, unit="batch")
                
                for batch_data_s4_train in train_pbar_s4_loop:
                    target_s4_b = batch_data_s4_train[0].to(CONFIG["device"])
                    s3_out_grid_b_for_s4_train = batch_data_s4_train[1].to(CONFIG["device"]) # Cond1 for S4
                    s4_new_feat_grid_b_for_s4_train = batch_data_s4_train[2].to(CONFIG["device"]) # Cond2 for S4

                    optimizer_s4.zero_grad()
                    t_s4_b = torch.randint(0, stage4_model.timesteps, (target_s4_b.shape[0],), device=CONFIG["device"]).long()
                    
                    condition_args_s4_loss = {
                        "stage3_output_grid_batch_for_s4": s3_out_grid_b_for_s4_train,
                        "stage4_new_condition_feature_grid_batch": s4_new_feat_grid_b_for_s4_train
                    }
                    loss_s4_batch = stage4_model.p_losses(
                        x_start_target_flow=target_s4_b, t=t_s4_b,
                        mode=ConditionMode.STAGE4, condition_args=condition_args_s4_loss
                    )
                    loss_s4_batch.backward()
                    optimizer_s4.step()
                    total_train_loss_epoch_s4 += loss_s4_batch.item()
                    train_pbar_s4_loop.set_postfix({"Batch Loss": f"{loss_s4_batch.item():.5f}"})

                avg_train_loss_epoch_s4 = total_train_loss_epoch_s4 / len(train_loader_s4_final) if len(train_loader_s4_final) > 0 else 0.0
                metrics_hist_s4['train_loss'].append(avg_train_loss_epoch_s4)

                # --- 驗證階段 (每個 Epoch 都執行) ---
                stage4_model.eval()
                total_val_loss_s4 = 0.0

                if val_loader_s4_final and hasattr(val_loader_s4_final, 'dataset') and len(val_loader_s4_final.dataset) > 0:
                    with torch.no_grad():
                        val_pbar_s4_loop = tqdm(val_loader_s4_final, desc=f"Epoch {epoch_s4_current} [S4 Validate]", leave=False, position=1)
                        for batch_data_s4_val in val_pbar_s4_loop:
                            target_s4_val_norm = batch_data_s4_val[0].to(CONFIG["device"])
                            s3_out_val_cond = batch_data_s4_val[1].to(CONFIG["device"])
                            s4_new_feat_val_cond = batch_data_s4_val[2].to(CONFIG["device"])

                            # 使用 p_losses 快速計算驗證損失
                            t_s4_val_b = torch.randint(0, stage4_model.timesteps, (target_s4_val_norm.shape[0],), device=CONFIG["device"]).long()
                            condition_args_s4_val = {
                                "stage3_output_grid_batch_for_s4": s3_out_val_cond,
                                "stage4_new_condition_feature_grid_batch": s4_new_feat_val_cond
                            }
                            val_loss_b_s4 = stage4_model.p_losses(
                                x_start_target_flow=target_s4_val_norm, t=t_s4_val_b,
                                mode=ConditionMode.STAGE4, condition_args=condition_args_s4_val
                            )
                            total_val_loss_s4 += val_loss_b_s4.item()
                    
                    avg_val_loss_s4 = total_val_loss_s4 / len(val_loader_s4_final)
                else:
                    avg_val_loss_s4 = float('inf') # 若驗證集為空，設為無效值

                metrics_hist_s4['val_loss'].append(avg_val_loss_s4)

                # --- 更新、日誌、儲存與早停 ---
                
                # 使用驗證損失來更新學習率排程器
                scheduler_s4.step(avg_val_loss_s4)
                current_lr_epoch_s4 = optimizer_s4.param_groups[0]['lr']
                metrics_hist_s4['lr'].append(current_lr_epoch_s4)

                val_loss_display_s4 = f"{avg_val_loss_s4:.5f}" if avg_val_loss_s4 != float('inf') else "N/A"

                # 更新主 epoch 進度條的後綴信息
                epoch_pbar_s4.set_postfix_str(f"Tr_Loss: {avg_train_loss_epoch_s4:.4f}, Val_Loss: {val_loss_display_s4}, LR: {current_lr_epoch_s4:.1e}, ES: {early_stopping_counter_s4}/{CONFIG.get('early_stopping_patience')}")

                # 使用驗證損失來判斷是否儲存最佳模型與早停
                if avg_val_loss_s4 < best_val_loss_s4:
                    best_val_loss_s4 = avg_val_loss_s4
                    early_stopping_counter_s4 = 0
                    tqdm.write(f"Epoch {epoch_s4_current}: 新最佳 Stage4 模型已儲存 (Val Loss: {best_val_loss_s4:.5f})。")
                    
                    torch.save({
                        'epoch': epoch_s4_current,
                        'ddpm_state_dict': stage4_model.state_dict(),
                        'optimizer_state_dict': optimizer_s4.state_dict(),
                        'scheduler_state_dict': scheduler_s4.state_dict(),
                        'best_val_loss_s4': best_val_loss_s4,
                        'config_snapshot_at_save': CONFIG,
                        'metrics_hist_s4': metrics_hist_s4,
                        'early_stopping_counter_s4': early_stopping_counter_s4,
                        's4_new_cond_feature_norm_stats': train_dataset_s4.norm_stats_s4_new_cond_feature, 
                        'stage4_avg_flow_map_dict': train_dataset_s4.average_flow_map_dict_current_stage,
                        'norm_stats_stage4_target': train_dataset_s4.norm_stats_current_stage_target
                    }, stage4_model_save_checkpoint_path_full)
                else:
                    early_stopping_counter_s4 +=1
                
                if early_stopping_counter_s4 >= CONFIG.get("early_stopping_patience"):
                    tqdm.write(f"Stage4 訓練因早停機制觸發於 Epoch {epoch_s4_current}。")
                    break
                    
            if 'epoch_pbar_s4' in locals() and isinstance(epoch_pbar_s4, tqdm):
                epoch_pbar_s4.close()
                
            logger.info(f"Stage4 模型 '{model_name_for_log}' 訓練完成。")
        else: # stage4_model is None (創建失敗)
            logger.error("Stage4 模型未能成功實例化，跳過訓練。")
    else: # train_loader_s4_final is None
        logger.info("跳過 Stage4 模型訓練，train_loader_s4_final 未定義或為空。")
        # 如果不訓練S4，嘗試載入預訓練的 Stage4 模型
        if not os.path.exists(CONFIG["stage4_checkpoint_full_path"]):
            logger.warning(f"Stage4 訓練被跳過，且未找到 Stage4 檢查點: {CONFIG['stage4_checkpoint_full_path']}")
            stage4_model = None 
        else:
            logger.info(f"從檢查點 {CONFIG['stage4_checkpoint_full_path']} 載入預訓練的 Stage4 模型...")
            chkpt_s4_load = torch.load(CONFIG["stage4_checkpoint_full_path"], map_location=CONFIG["device"], weights_only=False)
            cfg_s4_load = chkpt_s4_load.get('config_snapshot_at_save', CONFIG)
            unet_s4_load = UNet3D(cfg_s4_load.get("image_channels"), cfg_s4_load.get("base_channels_unet"), cfg_s4_load.get("time_emb_dim"), cfg_s4_load.get("condition_encode_dim"),dropout_rate=cfg_s4_load.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))).to(CONFIG["device"])
            stage4_model = DDPM3D(unet_s4_load, cfg_s4_load.get("timesteps"), (cfg_s4_load.get("D"), cfg_s4_load.get("H"), cfg_s4_load.get("W")), cfg_s4_load.get("image_channels"), cfg_s4_load.get("condition_input_channels"), cfg_s4_load.get("condition_encode_dim"), device=CONFIG["device"])
            stage4_model.load_state_dict(chkpt_s4_load['ddpm_state_dict'])
            logger.info(f"已載入預訓練的 Stage4 模型 (Epoch {chkpt_s4_load.get('epoch','未知')})。")

#%%
    if not (stage4_model and final_s3_model_to_eval and test_loader_s4_final):
        logger.warning("由於缺少 Stage4 模型、Stage3 模型或 Stage4 測試數據加載器，跳過最終評估。")
    else:
        # --- STAGE 4A: 一次性模型評估 (僅全地圖) ---
        # 這個區塊只會執行一次，計算所有模型的全地圖性能，這是最耗時的部分。
        logger.info(f"===== STAGE 4A: 開始一次性全地圖模型評估 =====")
        
        authoritative_s4_s3_metrics, authoritative_s4_s3_error_grids = None, None
        authoritative_baseline_metrics, authoritative_baseline_error_grids = None, None
        authoritative_s4_s3_predictions, authoritative_target_t_s4 = None, None
        authoritative_baseline_prediction, authoritative_target_t_baseline = None, None
        
        # 1. 準備 Inception 模型 (用於 FID)
        inception_model_for_fid_final_eval = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True).to(CONFIG["device"])
        inception_model_for_fid_final_eval.fc = nn.Identity()
        if hasattr(inception_model_for_fid_final_eval, 'AuxLogits'):
            inception_model_for_fid_final_eval.AuxLogits.fc = nn.Identity()
        inception_model_for_fid_final_eval.eval()

        # 2. 獲取正規化統計量
        stage4_target_norm_stats_for_eval = train_dataset_s4.norm_stats_current_stage_target
        stage3_target_norm_stats_for_eval = chkpt_s3_eval.get('norm_stats_stage3_target')
        if not stage4_target_norm_stats_for_eval or not stage3_target_norm_stats_for_eval:
            raise ValueError("無法獲取 Stage4 或 Stage3 目標的專用正規化統計量。")

        # 3. 執行 Stage4 vs Stage3 的評估
        logger.info("正在執行 Stage4 vs Stage3 的全地圖評估...")
        authoritative_s4_s3_metrics, authoritative_s4_s3_error_grids, authoritative_s4_s3_predictions, authoritative_target_t_s4 = evaluate_model(
            current_stage_model_trained=stage4_model,
            previous_stage_model_eval_instance=final_s3_model_to_eval,
            basemodel_eval_instance_for_s2_cond_generation=None,
            current_stage_mode=ConditionMode.STAGE4,
            dataloader_current_stage=test_loader_s4_final,
            inception_model_fid=inception_model_for_fid_final_eval,
            config=CONFIG,
            current_stage_target_norm_stats=stage4_target_norm_stats_for_eval,
            previous_stage_target_norm_stats=stage3_target_norm_stats_for_eval,
            target_grid_stds=global_target_grid_stds,
            target_overall_std=global_target_overall_std,
            max_samples_for_fid=CONFIG.get("fid_num_samples"),
            prefix="final_eval_full_map_S4_vs_S3_evaluation",
            grid_mask_hw=None,
            grid_mask_flat_indices=None
        )
        logger.info("Stage4 vs Stage3 全地圖評估完成。")

        # 4. 執行 Baseline 模型的獨立評估
        logger.info("===== 正在執行 Baseline 模型的全地圖評估 =====")
        baseline_model_path = CONFIG["baseline_model_path"]
        if not os.path.exists(baseline_model_path):
            logger.error(f"未找到 Baseline 模型檢查點: {baseline_model_path}，將跳過與 Baseline 的比較。")
            authoritative_baseline_metrics = {"baseline_model": {}}
            authoritative_baseline_error_grids = {"baseline_model": {}}
        else:
            chkpt_baseline = torch.load(baseline_model_path, map_location=CONFIG["device"], weights_only=False)
            cfg_baseline = chkpt_baseline.get('config_snapshot_at_save', chkpt_baseline.get('config'))
            unet_baseline = UNet3D(cfg_baseline["image_channels"], cfg_baseline["base_channels_unet"], cfg_baseline["time_emb_dim"], cfg_baseline["condition_encode_dim"], dropout_rate=cfg_baseline["unet_dropout_rate"]).to(CONFIG["device"])
            final_baseline_model_to_eval = DDPM3D(
                unet_model=unet_baseline, timesteps=cfg_baseline["timesteps"],
                image_size=(cfg_baseline["D"], cfg_baseline["H"], cfg_baseline["W"]),
                image_channels=cfg_baseline["image_channels"], condition_input_channels=cfg_baseline["condition_input_channels"],
                condition_encode_dim=cfg_baseline["condition_encode_dim"], device=CONFIG["device"]
            )
            final_baseline_model_to_eval.load_state_dict(chkpt_baseline['ddpm_state_dict'])
            logger.info(f"Baseline 模型載入完成。")

            baseline_test_dataset = BaselineDataset(
                df_for_processing=df_for_s4_specialist.iloc[s4_test_indices],
                config=CONFIG, mode='test',
                norm_stats_from_train=chkpt_baseline['cond_norm_stats'],
                target_info_from_train={"avg_flow_map": chkpt_baseline['target_avg_flow_map'], "norm_stats": chkpt_baseline['target_norm_stats']}
            )
            baseline_test_loader = DataLoader(baseline_test_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False)
            
            baseline_metrics_raw, baseline_error_grids_raw, authoritative_baseline_prediction, authoritative_target_t_baseline = evaluate_baseline_model_for_comparison(
                model_trained=final_baseline_model_to_eval, 
                dataloader=baseline_test_loader,
                inception_model_fid=inception_model_for_fid_final_eval, 
                config=CONFIG,
                target_norm_stats=chkpt_baseline['target_norm_stats'],
                target_grid_stds=global_target_grid_stds,
                target_overall_std=global_target_overall_std,
                prefix="final_eval_full_map_Baseline",
                grid_mask_hw=None,
                grid_mask_flat_indices=None
            )
            authoritative_baseline_metrics = {"baseline_model": baseline_metrics_raw}
            authoritative_baseline_error_grids = {"baseline_model": baseline_error_grids_raw}
            logger.info("Baseline 模型全地圖評估完成。")


        # --- STAGE 4B: 迴圈評估與報告產出 ---
        # 現在我們有了權威結果，進入迴圈來分別產生「全地圖」和「篩選後」的報告。
        logger.info(f"===== STAGE 4B: 開始基於已評估結果產出報告 =====")
        # 迴圈的意義: i=0 (全地圖), i=1 (原始篩選), i=2 (擴散篩選)
        for eval_mode in ['full_map', 'filtered', 'filtered_expanded']:
            
            logger.info(f"============================================================")
            logger.info(f"--- 開始處理第 {['full_map', 'filtered', 'filtered_expanded'].index(eval_mode) + 1} 輪報告: {eval_mode} ---")
            logger.info(f"============================================================")

            # --- 步驟 1: 根據不同模式，設定當前迴圈要使用的遮罩和檔名前綴 ---
            if eval_mode == 'full_map':
                current_prefix = "final_eval_full_map"
                current_mask_hw = None
                current_mask_flat = None
            
            elif eval_mode == 'filtered':
                if filtered_grid_mask_hw is None:
                    logger.warning("原始篩選器未啟用，跳過 'filtered' 報告產出。")
                    continue
                current_prefix = "final_eval_filtered"
                current_mask_hw = filtered_grid_mask_hw
                current_mask_flat = filtered_grid_indices_flat
            
            elif eval_mode == 'filtered_expanded':
                if expanded_grid_mask_hw is None:
                    logger.warning("擴散篩選器未產生 (因原始篩選器未啟用)，跳過 'filtered_expanded' 報告產出。")
                    continue
                current_prefix = "final_eval_filtered_expanded"
                current_mask_hw = expanded_grid_mask_hw
                current_mask_flat = expanded_grid_indices_flat
            
            # --- 步驟 2: 根據模式，準備指標 (直接使用或重新計算) ---
            final_s4_s3_metrics, final_baseline_metrics = None, None
            
            def recalculate_overall_metrics_from_tensors(pred_t, target_t, mask_hw, config, target_overall_std):
                if pred_t is None or target_t is None or mask_hw is None or pred_t.numel() == 0 or target_t.numel() == 0:
                    return {} # 如果沒有足夠的資料，返回空字典
                
                # 創建與目標張量形狀相同、設備相同的遮罩
                mask_tensor = torch.from_numpy(mask_hw.astype(bool)).to(target_t.device)
                mask_tensor = mask_tensor.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(target_t)

                # 使用遮罩篩選出感興趣的數據點 (會變成1D張量)
                pred_for_metric = torch.masked_select(pred_t, mask_tensor)
                target_for_metric = torch.masked_select(target_t, mask_tensor)
                
                epsilon = 1e-8
                mae = F.l1_loss(pred_for_metric, target_for_metric).item()
                actual_vals = torch.abs(target_for_metric)
                errors = torch.abs(target_for_metric - pred_for_metric)
                
                # 重新計算 MAPE Overall
                mape_overall_num = torch.sum(errors)
                mape_overall_den = torch.sum(actual_vals) + epsilon
                mape_overall = (mape_overall_num / mape_overall_den).item() * 100

                # 重新計算 SMAPE Overall
                smape_overall_den_sum = torch.sum(actual_vals + torch.abs(pred_for_metric))
                smape_overall = (200.0 * mape_overall_num / (smape_overall_den_sum + epsilon)).item()

                # 重新計算 STDE Overall
                # 注意: 這裡的分母應該是篩選後數據的標準差
                target_overall_std_filtered = torch.std(target_for_metric).item()
                stde_overall = mae / (target_overall_std_filtered + epsilon)

                return {
                    "mape_overall": mape_overall,
                    "smape_overall": smape_overall,
                    "stde_overall": stde_overall
                }
            def recalculate_scalar_metrics_from_grids(metrics_dict, error_grids_dict, mask_indices):
                    recalculated_metrics = {k: v.copy() for k, v in metrics_dict.items()}
                    if mask_indices is None: return recalculated_metrics
                    for model_name, grids in error_grids_dict.items():
                        if model_name in recalculated_metrics and isinstance(grids, dict):
                            key_map = {'MAE': 'mae', 'MSE': 'mse', 'MAPE': 'mape_avg_grid', 'SMAPE': 'smape_avg_grid', 'STDE_AvgGrid': 'stde_avg_grid'}
                            for metric_name_upper, key_lower in key_map.items():
                                if metric_name_upper in grids and grids[metric_name_upper] is not None:
                                    filtered_values = grids[metric_name_upper][mask_indices]
                                    new_avg_value = np.nanmean(filtered_values)
                                    recalculated_metrics[model_name][key_lower] = float(new_avg_value)
                    return recalculated_metrics

            if eval_mode == 'full_map':
                # 全地圖模式：直接複製權威結果
                logger.info("使用全地圖權威結果進行報告。")
                final_s4_s3_metrics = authoritative_s4_s3_metrics
                final_baseline_metrics = authoritative_baseline_metrics
                final_s4_s3_error_grids = authoritative_s4_s3_error_grids
                final_baseline_error_grids = authoritative_baseline_error_grids
            else:
                # 篩選模式 ('filtered' or 'filtered_expanded'): 基於權威 error_grids 和當前的遮罩重新計算
                logger.info(f"基於全地圖誤差網格和 '{eval_mode}' 遮罩，重新計算純量指標。")
                
                # 計算 AvgGrid 指標
                final_s4_s3_metrics = recalculate_scalar_metrics_from_grids(authoritative_s4_s3_metrics, authoritative_s4_s3_error_grids, current_mask_flat)
                final_baseline_metrics = recalculate_scalar_metrics_from_grids(authoritative_baseline_metrics, authoritative_baseline_error_grids, current_mask_flat)
                
                # 計算 Overall 指標
                logger.info(f"基於原始預測張量和 '{eval_mode}' 遮罩，重新計算 Overall 指標。")
                if authoritative_s4_s3_predictions and authoritative_target_t_s4 is not None:
                    for model_key, pred_tensor in authoritative_s4_s3_predictions.items():
                        if pred_tensor is not None and pred_tensor.numel() > 0 and model_key in final_s4_s3_metrics:
                            new_overall_metrics = recalculate_overall_metrics_from_tensors(pred_tensor, authoritative_target_t_s4, current_mask_hw, CONFIG, global_target_overall_std)
                            final_s4_s3_metrics[model_key].update(new_overall_metrics)
                
                if authoritative_baseline_prediction is not None and authoritative_target_t_baseline is not None:
                    new_baseline_overall = recalculate_overall_metrics_from_tensors(authoritative_baseline_prediction, authoritative_target_t_baseline, current_mask_hw, CONFIG, global_target_overall_std)
                    if "baseline_model" in final_baseline_metrics:
                        final_baseline_metrics["baseline_model"].update(new_baseline_overall)
                
                # 權威誤差圖在所有模式下都保持不變
                final_s4_s3_error_grids = authoritative_s4_s3_error_grids
                final_baseline_error_grids = authoritative_baseline_error_grids
                logger.info(f"'{eval_mode}' 模式指標重算完成。")

            # --- 報告產出邏輯 ---
            
            # 步驟 1: 合併所有原始指標
            logger.info("合併 Stage4, Stage3, 和 Baseline 的評估結果...")
            combined_metrics = {**final_s4_s3_metrics, **final_baseline_metrics}
            combined_error_grids = {**final_s4_s3_error_grids, **final_baseline_error_grids}

            # --- 步驟 2: 準備一個乾淨的、用於統一匯出的總體指標字典 ---
            summary_metrics_for_export = {}
            model_keys_for_summary = ['stage4_model', 'stage3_model_on_stage4_data', 'baseline_model']

            # 步驟 2.1: 先加入各個模型的指標
            for model_key in model_keys_for_summary:
                if model_key in combined_metrics and combined_metrics[model_key]:
                    # 【修改】在 round 之前，先用 float() 將 numpy 數值類型轉為 python 原生 float
                    summary_metrics_for_export[model_key] = {
                        k: round(float(v), 6) if isinstance(v, (float, np.number)) else v 
                        for k, v in combined_metrics[model_key].items()
                    }

            # 步驟 2.2: 計算並加入 Diff(S4-S3) 的差異指標
            if 'stage4_model' in summary_metrics_for_export and 'stage3_model_on_stage4_data' in summary_metrics_for_export:
                s4_m = summary_metrics_for_export['stage4_model']
                s3_m = summary_metrics_for_export['stage3_model_on_stage4_data']
                diff_s4_s3 = {}
                for key in s4_m.keys():
                    if key in s3_m and isinstance(s4_m.get(key), (int, float, np.number)) and isinstance(s3_m.get(key), (int, float, np.number)):
                         # 【修改】將計算結果也用 float() 轉換
                         diff_value = s4_m[key] - s3_m[key]
                         diff_s4_s3[key] = round(float(diff_value), 6)
                if diff_s4_s3:
                    summary_metrics_for_export['Diff(S4-S3)'] = diff_s4_s3

            # 步驟 2.3: 計算一次 Diff(S4-Baseline) 並加入
            if 'stage4_model' in summary_metrics_for_export and 'baseline_model' in summary_metrics_for_export:
                s4_m = summary_metrics_for_export['stage4_model']
                bl_m = summary_metrics_for_export['baseline_model']
                diff_s4_bl = {}
                for key in s4_m.keys():
                     if key in bl_m and isinstance(s4_m.get(key), (int, float, np.number)) and isinstance(bl_m.get(key), (int, float, np.number)):
                         # 【修改】將計算結果也用 float() 轉換
                         diff_value = s4_m[key] - bl_m[key]
                         diff_s4_bl[key] = round(float(diff_value), 6)
                if diff_s4_bl:
                    summary_metrics_for_export['Diff(S4-Baseline)'] = diff_s4_bl

            # --- 步驟 3. Log Info 並寫入 JSON 檔案 (此部分不變，但現在傳入的資料已安全) ---
            if summary_metrics_for_export:
                json_output_for_log = json.dumps(summary_metrics_for_export, indent=4, ensure_ascii=False)
                logger.info(f"準備寫入JSON的整體指標比較內容:\n{json_output_for_log}")

                json_path = os.path.join(CONFIG["stage4_model_save_dir"], f"{current_prefix}_comparison_summary.json")
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(summary_metrics_for_export, f, ensure_ascii=False, indent=4)
                    logger.info(f"整體指標比較已儲存至 JSON: {json_path}")
                except Exception as e:
                    logger.error(f"儲存 JSON 檔案失敗: {e}")

            # --- 步驟 4: 繪圖 ---
            if 'stage4_model' in combined_error_grids and 'baseline_model' in combined_error_grids and \
               combined_error_grids['stage4_model'] and combined_error_grids['baseline_model']:
                s4_err = combined_error_grids['stage4_model']
                bl_err = combined_error_grids['baseline_model']
                diff_s4_bl_grids = {}
                for metric in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
                    if s4_err.get(metric) is not None and bl_err.get(metric) is not None:
                        diff_s4_bl_grids[f"Diff_{metric}_(Stage4-Baseline)"] = s4_err[metric] - bl_err[metric]
                
                if diff_s4_bl_grids:
                    plot_grid_with_error_long_term(
                        dataset_for_coords=test_loader_s4_final.dataset,
                        error_metrics_grids=diff_s4_bl_grids,
                        config=CONFIG,
                        prefix=f"{current_prefix}_diff_S4_minus_Baseline",
                        grid_mask_flat_indices=current_mask_flat
                    )

            # --- 步驟 5: 準備並匯出 Excel 報告 ---
            excel_rows = []
            model_keys_in_report = ['stage4_model', 'stage3_model_on_stage4_data', 'baseline_model']
            num_grid_cells = CONFIG["H"] * CONFIG["W"]

            # 步驟 5.1: 寫入逐網格數據 (此部分不變)
            for model_key in model_keys_in_report:
                if model_key not in combined_error_grids or not combined_error_grids[model_key]: continue
                excel_rows.append({'資料來源': f"--- {model_key} (vs Stage4 Target) 逐網格誤差 ---"})
                error_grids = combined_error_grids[model_key]
                indices_to_loop = current_mask_flat if is_filtered_eval and current_mask_flat is not None else range(num_grid_cells)
                for flat_idx in indices_to_loop:
                    row_data = {'資料來源': model_key, '網格座標_R': flat_idx // CONFIG["W"], '網格座標_C': flat_idx % CONFIG["W"]}
                    for metric in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
                        if metric in error_grids and error_grids[metric] is not None:
                            row_data[metric] = error_grids[metric][flat_idx]
                        else:
                            row_data[metric] = np.nan
                    if 'STDE_AvgGrid' in error_grids and error_grids['STDE_AvgGrid'] is not None:
                        row_data['STDE_AvgGrid'] = error_grids['STDE_AvgGrid'][flat_idx]
                    else:
                        row_data['STDE_AvgGrid'] = np.nan
                    excel_rows.append(row_data)

            # 步驟 5.2: 寫入逐網格誤差差異 (此部分不變)
            if 'stage4_model' in combined_error_grids and 'stage3_model_on_stage4_data' in combined_error_grids and \
            combined_error_grids['stage4_model'] and combined_error_grids['stage3_model_on_stage4_data']:
                s4_s3_diff_grids = {}
                excel_rows.append({'資料來源': "--- Difference (Stage4 - Stage3) 逐網格誤差 ---"})
                s4_err = combined_error_grids['stage4_model']
                s3_err = combined_error_grids['stage3_model_on_stage4_data']
                for metric in ['MSE', 'MAE', 'MAPE', 'SMAPE', 'STDE_AvgGrid']:
                    if metric in s4_err and metric in s3_err and s4_err[metric] is not None and s3_err[metric] is not None:
                        s4_s3_diff_grids[metric] = s4_err[metric] - s3_err[metric]

                indices_to_loop = current_mask_flat if is_filtered_eval and current_mask_flat is not None else range(num_grid_cells)
                for flat_idx in indices_to_loop:
                    row_data = {'資料來源': "Diff(S4-S3)", '網格座標_R': flat_idx // CONFIG["W"], '網格座標_C': flat_idx % CONFIG["W"]}
                    for metric in ['MSE', 'MAE', 'MAPE', 'SMAPE', 'STDE_AvgGrid']:
                        if metric in s4_s3_diff_grids and s4_s3_diff_grids[metric] is not None:
                            row_data[metric] = s4_s3_diff_grids[metric][flat_idx]
                        else:
                            row_data[metric] = np.nan
                    excel_rows.append(row_data)

            if 'stage4_model' in combined_error_grids and 'baseline_model' in combined_error_grids and \
            combined_error_grids['stage4_model'] and combined_error_grids['baseline_model']:
                excel_rows.append({'資料來源': "--- Difference (Stage4 - Baseline) 逐網格誤差 ---"})
                s4_err = combined_error_grids['stage4_model']
                bl_err = combined_error_grids['baseline_model']
                indices_to_loop = current_mask_flat if is_filtered_eval and current_mask_flat is not None else range(num_grid_cells)
                for flat_idx in indices_to_loop:
                    row_data = {'資料來源': "Diff(S4-Baseline)", '網格座標_R': flat_idx // CONFIG["W"], '網格座標_C': flat_idx % CONFIG["W"]}
                    for metric in ['MSE', 'MAE', 'MAPE', 'SMAPE', 'STDE_AvgGrid']:
                        if metric in s4_err and metric in bl_err and s4_err[metric] is not None and bl_err[metric] is not None:
                            row_data[metric] = s4_err[metric][flat_idx] - bl_err[metric][flat_idx]
                        else:
                            row_data[metric] = np.nan
                    excel_rows.append(row_data)

            # 步驟 5.3: 【修改】寫入總體平均指標 (共用 summary_metrics_for_export 的結果)
            excel_rows.append({'資料來源': f"--- 整體指標比較 ---"})
            key_to_excel_header_map = {
                'mse': 'MSE', 'mae': 'MAE',
                'mape_avg_grid': 'MAPE(AvgGrid)', 'smape_avg_grid': 'SMAPE(AvgGrid)',
                'stde_avg_grid': 'STDE_AvgGrid',
                'mape_overall': 'MAPE(Overall)', 'smape_overall': 'SMAPE(Overall)',
                'stde_overall': 'STDE_Overall',
                'fid': 'FID'
            }
            # 定義匯出順序
            export_order = ['stage4_model', 'stage3_model_on_stage4_data', 'baseline_model', 'Diff(S4-S3)', 'Diff(S4-Baseline)']
            
            for source_name in export_order:
                if source_name in summary_metrics_for_export:
                    metrics = summary_metrics_for_export[source_name]
                    is_diff_row = 'Diff' in source_name
                    row_label = "整體平均差異" if is_diff_row else "整體平均"
                    
                    avg_row = {'資料來源': source_name, '網格座標_R': row_label}
                    for key, header in key_to_excel_header_map.items():
                        avg_row[header] = metrics.get(key)
                    
                    excel_rows.append(avg_row)

            # 步驟 5.4: 匯出到 Excel 檔案
            df_export = pd.DataFrame(excel_rows)
            excel_path = os.path.join(CONFIG["stage4_model_save_dir"], f"{current_prefix}_comparison_report.xlsx")
            df_export.to_excel(excel_path, index=False, sheet_name="Full_Comparison")
            logger.info(f"報告已匯出至 Excel: {excel_path}")

    logger.info(f"===== DDPM 多階段流程 (含Stage4) 結束 =====")
# %%

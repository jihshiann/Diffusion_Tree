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
    "base_channels_unet": 64,   # UNet3D 的基礎通道數
    "unet_dropout_rate": 0.1,
    "time_emb_dim": 256,        # 時間嵌入維度
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 / UNet中與x_t合併的維度

    # === Basemodel 相關 (用於載入並決定其原始條件處理方式) ===
    # Basemodel 的 condition_processor 輸入通道數 (通常是2，因為它內部將小時、假日轉為2個網格)
    "basemodel_condition_input_channels": 2, # 假設原始basemodel用2通道條件(小時網格+假日網格)
    "basemodel_checkpoint_to_load_for_stage2": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_long-term\best_ddpm_model_during_training.pth", # Basemodel檢查點

    # === Stage2 特定配置 ===
    "stage2_new_condition_feature_column": "月", # Stage2 新條件的欄位名
    "stage2_new_conditional_operator": "==",         # Stage2 新條件的運算符
    "stage2_new_conditional_value": 4,             # Stage2 新條件的閾值
    "stage2_model_name": "Stage2_MonthE4",    # 第二階段模型的名稱
    "stage2_ddpm_condition_input_channels": 2,       # Stage2 DDPM 的 condition_processor 輸入通道數 (固定為2: bm_out + uv_grid_s2)
    "stage2_checkpoint_path": "best_stage2_model_Month_E_4.pth", # Stage2 模型的檢查點檔名 (相對路徑)
    "basemodel_checkpoint_to_load_for_stage3": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage2\Stage2_MonthE4\best_stage2_model_Month_E_4.pth", # Stage2檢查點 (用於載入給Stage3)

    # === Stage3 特定配置 ===
    "stage3_new_condition_feature_column": "日", # Stage3 新條件的欄位名 
    "stage3_new_conditional_operator": "<=",         # Stage3 新條件的運算符
    "stage3_new_conditional_value": 7,             # Stage3 新條件的閾值 (例如: 4 代表週一到週五)
    "stage3_model_name": "Stage3_Dayle7",    # 第三階段模型的名稱
    "stage3_ddpm_condition_input_channels": 2,       # Stage3 DDPM 的 condition_processor 輸入通道數 (固定為2: s2_out + uv_grid_s3)
    "stage3_checkpoint_path": "best_stage3_model_Day_le_7.pth", # Stage3 模型的檢查點檔名 (相對路徑)

    # --- DDPM 擴散參數 ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 訓練參數 (Stage2/Stage3 將優先使用 epochs_stageX, lr_stageX 等，若無則回退到通用版本) ---
    "epochs": 128,
    "batch_size": 128,
    "lr": 1e-3,

    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "weight_decay": 1e-5,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 3,
    "lr_scheduler_min_lr": 1e-7,
    "early_stopping_patience": 6,
    "val_calculation_freq": 4,

    # --- 評估參數 ---
    "eval_batch_size": 256,
    "fid_batch_size": 256,
    "fid_num_samples": 256,

    # --- 路徑與儲存 ---
    "save_dir": "results_ddpm_stage3", # 主結果儲存目錄的基礎名稱
    
    "train_split_ratio": 0.7,
    "val_split_ratio": 0.15,

    # --- 快取設定 ---
    "cache_dir_name": "model_outputs_cache", # 相對於 save_dir 的快取目錄名稱
    "cached_basemodel_outputs_for_s2_filename": "basemodel_outputs_for_s2_normalized.npy",
    "cached_stage2_outputs_for_s3_filename": "stage2_outputs_for_s3_normalized.npy",
}

# 根據當前活躍的最高階段來設定通用的 condition_input_channels
# 這主要影響模型實例化時 DDPM3D 的 condition_processor。
# 在訓練/採樣時，我們會明確傳遞該階段所需的條件網格數量。
# 這裡假設每個階段的 DDPM condition_processor 都期望2個輸入通道。
CONFIG["condition_input_channels"] = CONFIG.get("stage3_ddpm_condition_input_channels", 
                                            CONFIG.get("stage2_ddpm_condition_input_channels", 2))


# 更新/生成 Stage2 相關路徑
CONFIG["stage2_model_save_dir"] = os.path.join(CONFIG["save_dir"], CONFIG["stage2_model_name"])
os.makedirs(CONFIG["stage2_model_save_dir"], exist_ok=True)
CONFIG["stage2_checkpoint_full_path"] = os.path.join(CONFIG["stage2_model_save_dir"], CONFIG["stage2_checkpoint_path"])

# 更新/生成 Stage3 相關路徑
CONFIG["stage3_model_save_dir"] = os.path.join(CONFIG["save_dir"], CONFIG["stage3_model_name"])
os.makedirs(CONFIG["stage3_model_save_dir"], exist_ok=True)
CONFIG["stage3_checkpoint_full_path"] = os.path.join(CONFIG["stage3_model_save_dir"], CONFIG["stage3_checkpoint_path"])

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
logger.info(f"Stage2 結果將儲存於: {CONFIG['stage2_model_save_dir']}")

if not os.path.exists(CONFIG["basemodel_checkpoint_to_load_for_stage2"]):
    logger.error(f"【【【警告】】】 Basemodel 檢查點路徑未設定或檔案不存在: {CONFIG['basemodel_checkpoint_to_load_for_stage2']}")
if not os.path.exists(CONFIG["basemodel_checkpoint_to_load_for_stage3"]): # 檢查 Stage2 檢查點路徑
    logger.error(f"【【【警告】】】 Stage2 檢查點路徑 (用於載入給Stage3) 未設定或檔案不存在: {CONFIG['basemodel_checkpoint_to_load_for_stage3']}")
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

    def _prepare_stage2_condition_grids(self,
                                     condition_grid_1_batch: torch.Tensor,
                                     condition_grid_2_batch: torch.Tensor
                                     ) -> torch.Tensor:
        expected_single_grid_shape = (1, self.image_size_D, self.image_size_H, self.image_size_W)
        # 檢查第一個條件網格的通道數是否為1 (因為通常是單一來源的網格，如BM輸出或S2輸出)
        if condition_grid_1_batch.shape[1] != 1:
            self.logger.warning(f"Stage2/3 condition_grid_1_batch has {condition_grid_1_batch.shape[1]} channels, expected 1. Using as is.")
            # Consider raising error or taking first channel if strictly 1 channel is expected before cat.
        # 檢查第二個條件網格的通道數是否為1 (因為通常是單一來源的網格，如新特徵網格)
        if condition_grid_2_batch.shape[1] != 1:
            self.logger.warning(f"Stage2/3 condition_grid_2_batch has {condition_grid_2_batch.shape[1]} channels, expected 1. Using as is.")

        # 確保空間維度 (D, H, W) 匹配
        if condition_grid_1_batch.shape[2:] != expected_single_grid_shape[1:] or \
           condition_grid_2_batch.shape[2:] != expected_single_grid_shape[1:]:
            self.logger.error(f"Stage 2/3 condition input grid spatial dimensions (D,H,W) are incorrect or mismatched. "
                              f"Grid1 spatial: {condition_grid_1_batch.shape[2:]}, Grid2 spatial: {condition_grid_2_batch.shape[2:]}. "
                              f"Expected spatial: {expected_single_grid_shape[1:]}")
            # Consider raising an error.
        
        # 這裡假設 condition_processor 的輸入通道數是 2
        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_stage2_condition_grids (used by Stage2/3): Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels by concatenating two 1-channel grids.")
        return torch.cat((condition_grid_1_batch, condition_grid_2_batch), dim=1)

    def p_losses(self, x_start_target_flow: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None,
                 # 條件參數 - 擇一組提供
                 # Basemodel 條件
                 hour_scalars_batch: Optional[torch.Tensor] = None,
                 is_holiday_scalars_batch: Optional[torch.Tensor] = None,
                 # Stage2 條件
                 basemodel_output_grid_batch: Optional[torch.Tensor] = None,
                 stage2_new_condition_feature_grid_batch: Optional[torch.Tensor] = None,
                 # Stage3 條件
                 stage2_output_grid_batch_for_s3: Optional[torch.Tensor] = None,
                 stage3_new_condition_feature_grid_batch: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:

        if noise is None: noise = torch.randn_like(x_start_target_flow)
        x_t_noisy_target = self.q_sample(x_start=x_start_target_flow, t=t, noise=noise)

        stacked_cond_grids: Optional[torch.Tensor] = None
        # 判斷是哪種條件模式
        if hour_scalars_batch is not None and is_holiday_scalars_batch is not None:
            # Basemodel (原始) 條件模式
            if basemodel_output_grid_batch is not None or stage2_new_condition_feature_grid_batch is not None or \
               stage2_output_grid_batch_for_s3 is not None or stage3_new_condition_feature_grid_batch is not None:
                raise ValueError("p_losses (Basemodel mode): Cannot provide scalar (hour/holiday) and other grid conditions simultaneously.")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(
                hour_scalars_batch, is_holiday_scalars_batch
            )
        elif basemodel_output_grid_batch is not None and stage2_new_condition_feature_grid_batch is not None:
            # Stage2 條件模式
            if hour_scalars_batch is not None or is_holiday_scalars_batch is not None or \
               stage2_output_grid_batch_for_s3 is not None or stage3_new_condition_feature_grid_batch is not None:
                raise ValueError("p_losses (Stage2 mode): Cannot provide Stage2 grid conditions and other mode conditions simultaneously.")
            stacked_cond_grids = self._prepare_stage2_condition_grids( # Reusing this method
                basemodel_output_grid_batch,
                stage2_new_condition_feature_grid_batch
            )
        elif stage2_output_grid_batch_for_s3 is not None and stage3_new_condition_feature_grid_batch is not None:
            # Stage3 條件模式
            if hour_scalars_batch is not None or is_holiday_scalars_batch is not None or \
               basemodel_output_grid_batch is not None or stage2_new_condition_feature_grid_batch is not None:
                raise ValueError("p_losses (Stage3 mode): Cannot provide Stage3 grid conditions and other mode conditions simultaneously.")
            stacked_cond_grids = self._prepare_stage2_condition_grids( # Reusing this method
                stage2_output_grid_batch_for_s3,
                stage3_new_condition_feature_grid_batch
            )
        else:
            raise ValueError("p_losses: Insufficient or ambiguous condition arguments provided for any mode.")

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
               # Basemodel 條件
               hour_scalars_batch: Optional[torch.Tensor] = None,
               is_holiday_scalars_batch: Optional[torch.Tensor] = None,
               # Stage2 條件
               basemodel_output_grid_batch: Optional[torch.Tensor] = None,
               stage2_new_condition_feature_grid_batch: Optional[torch.Tensor] = None,
               # Stage3 條件
               stage2_output_grid_batch_for_s3: Optional[torch.Tensor] = None,
               stage3_new_condition_feature_grid_batch: Optional[torch.Tensor] = None
               ) -> torch.Tensor:

        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)

        stacked_cond_grids: Optional[torch.Tensor] = None
        if hour_scalars_batch is not None and is_holiday_scalars_batch is not None:
            # Basemodel (原始) 條件模式
            if basemodel_output_grid_batch is not None or stage2_new_condition_feature_grid_batch is not None or \
               stage2_output_grid_batch_for_s3 is not None or stage3_new_condition_feature_grid_batch is not None:
                raise ValueError("sample (Basemodel mode): Cannot provide scalar (hour/holiday) and other grid conditions simultaneously.")
            if hour_scalars_batch.shape[0] != batch_size or is_holiday_scalars_batch.shape[0] != batch_size:
                raise ValueError(f"Original condition batch sizes ({hour_scalars_batch.shape[0]},{is_holiday_scalars_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(
                hour_scalars_batch, is_holiday_scalars_batch
            ).to(self.device)
        elif basemodel_output_grid_batch is not None and stage2_new_condition_feature_grid_batch is not None:
            # Stage2 條件模式
            if hour_scalars_batch is not None or is_holiday_scalars_batch is not None or \
               stage2_output_grid_batch_for_s3 is not None or stage3_new_condition_feature_grid_batch is not None:
                raise ValueError("sample (Stage2 mode): Cannot provide Stage2 grid conditions and other mode conditions simultaneously.")
            if basemodel_output_grid_batch.shape[0] != batch_size or stage2_new_condition_feature_grid_batch.shape[0] != batch_size:
                raise ValueError(f"Stage2 condition batch sizes ({basemodel_output_grid_batch.shape[0]},{stage2_new_condition_feature_grid_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_stage2_condition_grids( # Reusing
                basemodel_output_grid_batch,
                stage2_new_condition_feature_grid_batch
            )
        elif stage2_output_grid_batch_for_s3 is not None and stage3_new_condition_feature_grid_batch is not None:
            # Stage3 條件模式
            if hour_scalars_batch is not None or is_holiday_scalars_batch is not None or \
               basemodel_output_grid_batch is not None or stage2_new_condition_feature_grid_batch is not None:
                raise ValueError("sample (Stage3 mode): Cannot provide Stage3 grid conditions and other mode conditions simultaneously.")
            if stage2_output_grid_batch_for_s3.shape[0] != batch_size or stage3_new_condition_feature_grid_batch.shape[0] != batch_size:
                raise ValueError(f"Stage3 condition batch sizes ({stage2_output_grid_batch_for_s3.shape[0]},{stage3_new_condition_feature_grid_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_stage2_condition_grids( # Reusing
                stage2_output_grid_batch_for_s3,
                stage3_new_condition_feature_grid_batch
            )
        else:
            raise ValueError("sample: Insufficient or ambiguous condition arguments provided for any mode.")

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
    
def create_stage2_model_from_basemodel_checkpoint(
                               basemodel_checkpoint_path: str,
                               config_for_stage2_model: Dict[str, Any],
                               device: str
                               ) -> DDPM3D:
    logger.info(f"從 Basemodel 檢查點 {basemodel_checkpoint_path} 創建並初始化 Stage2 模型...")
    
    chkpt_basemodel = torch.load(basemodel_checkpoint_path, map_location=device, weights_only = False)
    if 'ddpm_state_dict' not in chkpt_basemodel:
        raise KeyError(f"Basemodel 檢查點 {basemodel_checkpoint_path} 中未找到 'ddpm_state_dict'。")
    if 'selected_sensor_info' not in chkpt_basemodel: # 確保 selected_sensor_info 存在
        raise KeyError(f"'selected_sensor_info' 不存在於 Basemodel 檢查點 {basemodel_checkpoint_path} 中。")
    
    basemodel_original_config = chkpt_basemodel.get('config', config_for_stage2_model)

    # Stage2 模型的 UNet 架構應與 Basemodel 一致
    stage2_unet = UNet3D(
        input_image_channels=basemodel_original_config.get("image_channels", config_for_stage2_model["image_channels"]),
        base_channels=basemodel_original_config.get("base_channels_unet", config_for_stage2_model["base_channels_unet"]),
        time_emb_dim=basemodel_original_config.get("time_emb_dim", config_for_stage2_model["time_emb_dim"]),
        condition_encode_dim=basemodel_original_config.get("condition_encode_dim", config_for_stage2_model["condition_encode_dim"]),
        dropout_rate=basemodel_original_config.get("unet_dropout_rate", config_for_stage2_model.get("unet_dropout_rate", 0.05))
    ).to(device)

    stage2_model_condition_input_channels = config_for_stage2_model.get("condition_input_channels", CONFIG.get("stage2_ddpm_condition_input_channels", 2))
    logger.info(f"Stage2 模型將使用 {stage2_model_condition_input_channels} 個條件輸入通道。")

    stage2_model_instance = DDPM3D(
        unet_model=stage2_unet,
        timesteps=config_for_stage2_model["timesteps"],
        image_size=(config_for_stage2_model["D"], config_for_stage2_model["H"], config_for_stage2_model["W"]),
        image_channels=config_for_stage2_model["image_channels"],
        condition_input_channels=stage2_model_condition_input_channels,
        condition_encode_dim=config_for_stage2_model["condition_encode_dim"],
        beta_start=config_for_stage2_model["beta_start"],
        beta_end=config_for_stage2_model["beta_end"],
        device=device
    )

    logger.info(f"將 Basemodel 的權重載入到新的 Stage2 模型實例 (condition_input_channels={stage2_model_condition_input_channels})...")
    try:
        # 嘗試載入完整的 state_dict
        # 如果 Stage2 模型的 condition_processor 與 Basemodel 的不同 (例如，不同的 condition_input_channels 導致不同的 Conv3d 層)，
        # 這裡可能會失敗。
        stage2_model_instance.load_state_dict(chkpt_basemodel['ddpm_state_dict'])
        logger.info("Stage2 模型權重從 Basemodel 完整遷移完成。")
    except RuntimeError as e:
        logger.warning(f"直接載入 Basemodel state_dict 到 Stage2 模型失敗: {e}")
        logger.warning("這可能是因為 Stage2 模型的 condition_processor 與 Basemodel 的不同 (例如不同的輸入通道數)。")
        logger.warning("嘗試僅載入 UNet (model) 部分的權重，並重新初始化 condition_processor...")
        
        # 載入 UNet 權重
        unet_state_dict = {k.replace('model.', ''): v for k, v in chkpt_basemodel['ddpm_state_dict'].items() if k.startswith('model.')}
        stage2_model_instance.model.load_state_dict(unet_state_dict)
        logger.info("僅 UNet 權重從 Basemodel 遷移完成。")

        # 重新初始化 Stage2 模型的 condition_processor
        # 確保 condition_processor 使用 Stage2 配置的輸入通道數
        stage2_cond_input_ch = config_for_stage2_model.get("condition_input_channels", CONFIG.get("stage2_ddpm_condition_input_channels", 2))
        stage2_cond_encode_dim = config_for_stage2_model.get("condition_encode_dim", CONFIG.get("condition_encode_dim"))
        
        stage2_model_instance.condition_processor = nn.Sequential(
            nn.Conv3d(stage2_cond_input_ch, stage2_cond_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(stage2_cond_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(stage2_cond_encode_dim // 2, stage2_cond_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(stage2_cond_encode_dim), nn.SiLU()
        ).to(device)
        logger.info(f"Stage2 模型的 condition_processor 已使用 {stage2_cond_input_ch} 輸入通道重新初始化。")

    return stage2_model_instance

# --------------------------------------
# Stage3 模型創建函數
# --------------------------------------
def create_stage3_model_from_stage2_checkpoint(
                               stage2_checkpoint_path: str,
                               config_for_stage3_model: Dict[str, Any],
                               device: str
                               ) -> DDPM3D:
    logger.info(f"從 Stage2 檢查點 {stage2_checkpoint_path} 創建並初始化 Stage3 模型...")
    
    chkpt_stage2 = torch.load(stage2_checkpoint_path, map_location=device, weights_only=False)
    if 'ddpm_state_dict' not in chkpt_stage2:
        raise KeyError(f"Stage2 檢查點 {stage2_checkpoint_path} 中未找到 'ddpm_state_dict'。")
    
    # Stage3 模型的 UNet 架構應與 Stage2 (以及 Basemodel) 一致
    # 從 Stage2 檢查點的 config 或傳入的 config_for_stage3_model 獲取 UNet 參數
    s2_chkpt_config = chkpt_stage2.get('config_snapshot_at_save', config_for_stage3_model)

    stage3_unet = UNet3D(
        input_image_channels=s2_chkpt_config.get("image_channels", config_for_stage3_model["image_channels"]),
        base_channels=s2_chkpt_config.get("base_channels_unet", config_for_stage3_model["base_channels_unet"]),
        time_emb_dim=s2_chkpt_config.get("time_emb_dim", config_for_stage3_model["time_emb_dim"]),
        condition_encode_dim=s2_chkpt_config.get("condition_encode_dim", config_for_stage3_model["condition_encode_dim"]),
        dropout_rate=s2_chkpt_config.get("unet_dropout_rate", config_for_stage3_model.get("unet_dropout_rate", 0.05))
    ).to(device)

    # Stage3 模型的 condition_processor 輸入通道數
    stage3_model_condition_input_channels = config_for_stage3_model.get("condition_input_channels", CONFIG.get("stage3_ddpm_condition_input_channels", 2))
    logger.info(f"Stage3 模型將使用 {stage3_model_condition_input_channels} 個條件輸入通道。")

    stage3_model_instance = DDPM3D(
        unet_model=stage3_unet,
        timesteps=config_for_stage3_model["timesteps"],
        image_size=(config_for_stage3_model["D"], config_for_stage3_model["H"], config_for_stage3_model["W"]),
        image_channels=config_for_stage3_model["image_channels"],
        condition_input_channels=stage3_model_condition_input_channels, # 使用 Stage3 的配置
        condition_encode_dim=config_for_stage3_model["condition_encode_dim"],
        beta_start=config_for_stage3_model["beta_start"],
        beta_end=config_for_stage3_model["beta_end"],
        device=device
    )

    logger.info(f"將 Stage2 模型的權重載入到新的 Stage3 模型實例 (condition_input_channels={stage3_model_condition_input_channels})...")
    try:
        stage3_model_instance.load_state_dict(chkpt_stage2['ddpm_state_dict'])
        logger.info("Stage3 模型權重從 Stage2 完整遷移完成。")
    except RuntimeError as e:
        logger.warning(f"直接載入 Stage2 state_dict 到 Stage3 模型失敗: {e}")
        logger.warning("這可能是因為 Stage3 模型的 condition_processor 與 Stage2 的不同 (例如不同的輸入通道數)。")
        logger.warning("嘗試僅載入 UNet (model) 部分的權重，並重新初始化 condition_processor...")
        
        unet_state_dict_s3 = {k.replace('model.', ''): v for k, v in chkpt_stage2['ddpm_state_dict'].items() if k.startswith('model.')}
        stage3_model_instance.model.load_state_dict(unet_state_dict_s3)
        logger.info("僅 UNet 權重從 Stage2 遷移完成。")

        s3_cond_input_ch = config_for_stage3_model.get("condition_input_channels", CONFIG.get("stage3_ddpm_condition_input_channels", 2))
        s3_cond_encode_dim = config_for_stage3_model.get("condition_encode_dim", CONFIG.get("condition_encode_dim"))
        
        stage3_model_instance.condition_processor = nn.Sequential(
            nn.Conv3d(s3_cond_input_ch, s3_cond_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(s3_cond_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(s3_cond_encode_dim // 2, s3_cond_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(s3_cond_encode_dim), nn.SiLU()
        ).to(device)
        logger.info(f"Stage3 模型的 condition_processor 已使用 {s3_cond_input_ch} 輸入通道重新初始化。")

    return stage3_model_instance

# --------------------------------------
# 數據處理相關
# --------------------------------------
def parse_lat_lon(column_name: str) -> tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class Stage2Dataset(Dataset):
    def __init__(self,
                 df_for_stage2_processing: pd.DataFrame,
                 basemodel_outputs_for_samples_np: np.ndarray, # BM 正規化後的輸出 (N, 1, D, H, W)
                 config: Dict[str, Any],
                 original_sorted_flow_columns: List[str],
                 mode: str = 'train',
                 # 從訓練集傳遞給驗證/測試集的統計數據
                 stage3_avg_flow_map_dict_from_train: Optional[Dict[Tuple, np.ndarray]] = None,
                 s2_new_cond_feature_norm_stats_from_s2_train: Optional[Dict[str, float]] = None, # Stage2 新條件的統計數據
                 s3_new_cond_feature_norm_stats_from_train: Optional[Dict[str, float]] = None,   # Stage3 新條件的統計數據
                 stage3_target_norm_stats_from_train: Optional[Dict[str, float]] = None
                 ):
        super().__init__()
        self.df_s2 = df_for_stage2_processing.reset_index(drop=True)
        self.basemodel_outputs_np = basemodel_outputs_for_samples_np # (N, 1, D, H, W) - 正規化
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.Stage2Dataset")

        self.H = config["H"]
        self.W = config["W"]
        self.D = config.get("D", 1)
        self.image_channels_target = config.get("image_channels", 1)
        self.sorted_flow_columns = original_sorted_flow_columns

        # --- 處理 Basemodel 的原始條件 (小時, 假日) ---
        if '時' not in self.df_s2.columns: raise KeyError("DataFrame 中找不到 '時' 欄位 (for Stage2Dataset BM conditions)。")
        dt_series_for_base_cond = pd.to_datetime(self.df_s2['時'])
        self.hours_for_target_np = dt_series_for_base_cond.dt.hour.values
        self.hour_category_for_target_grouping_np = (self.hours_for_target_np > 8).astype(int)

        if 'holiday' not in self.df_s2.columns and 'hoilday' in self.df_s2.columns: self.df_s2.rename(columns={"hoilday": "holiday"}, inplace=True)
        if 'holiday' not in self.df_s2.columns: raise KeyError("DataFrame 中找不到 'holiday' 或 'hoilday' 欄位 (for Stage2Dataset BM conditions)。")
        if self.df_s2['holiday'].dtype == bool: self.is_holiday_for_target_np = self.df_s2['holiday'].astype(int).values
        elif pd.api.types.is_numeric_dtype(self.df_s2['holiday']): self.is_holiday_for_target_np = self.df_s2['holiday'].fillna(0).astype(bool).astype(int).values
        else:
            holiday_map = {'是': 1, 'true': 1, '1': 1, 'yes': 1, 'y': 1, '否': 0, 'false': 0, '0': 0, 'no': 0, 'n': 0}
            self.is_holiday_for_target_np = self.df_s2['holiday'].astype(str).str.lower().map(holiday_map).fillna(0).astype(int).values
        self.logger.info(f"Stage2Dataset (mode={self.mode}): BM 條件 (小時, 假日) 處理完畢。")

        # --- 處理 Stage2 的新條件 ---
        self.s2_new_cond_col_name = config["stage2_new_condition_feature_column"]
        self.s2_new_cond_op = config["stage2_new_conditional_operator"]
        self.s2_new_cond_val = config["stage2_new_conditional_value"]
        if self.s2_new_cond_col_name not in self.df_s2.columns: raise ValueError(f"Stage2Dataset: Stage2 的條件欄位 '{self.s2_new_cond_col_name}' 不在 DataFrame 中。")
        self.s2_new_cond_original_values_np = pd.to_numeric(self.df_s2[self.s2_new_cond_col_name], errors='coerce').values
        self.s2_new_cond_category_for_target_np = self._calculate_category_vector(
            self.s2_new_cond_original_values_np, self.s2_new_cond_op, self.s2_new_cond_val, self.s2_new_cond_col_name, "Stage2Cond"
        )
        self.logger.info(f"Stage2Dataset (mode={self.mode}): Stage2 條件 ('{self.s2_new_cond_col_name}') 分類處理完畢。")

        # --- 處理 Stage3 的新條件 ---
        self.s3_new_cond_col_name = config["stage3_new_condition_feature_column"]
        self.s3_new_cond_op = config["stage3_new_conditional_operator"]
        self.s3_new_cond_val = config["stage3_new_conditional_value"]
        if self.s3_new_cond_col_name not in self.df_s2.columns: raise ValueError(f"Stage2Dataset: Stage3 的條件欄位 '{self.s3_new_cond_col_name}' 不在 DataFrame 中。")
        self.s3_new_cond_original_values_np = pd.to_numeric(self.df_s2[self.s3_new_cond_col_name], errors='coerce').values
        self.s3_new_cond_category_for_target_np = self._calculate_category_vector(
            self.s3_new_cond_original_values_np, self.s3_new_cond_op, self.s3_new_cond_val, self.s3_new_cond_col_name, "Stage3Cond"
        )
        self.logger.info(f"Stage2Dataset (mode={self.mode}): Stage3 條件 ('{self.s3_new_cond_col_name}') 分類處理完畢。")

        # --- 根據模式處理正規化統計量和平均流量圖 ---
        if self.mode == 'train':
            # S2 新條件特徵的正規化統計 (如果 Stage2Dataset 沒傳，理論上應該由 Stage2Dataset 計算並傳入)
            if s2_new_cond_feature_norm_stats_from_s2_train:
                self.norm_stats_s2_new_cond_feature = s2_new_cond_feature_norm_stats_from_s2_train
                self.s2_new_cond_feature_mean = self.norm_stats_s2_new_cond_feature.get('mean', 0.0)
                self.s2_new_cond_feature_std = self.norm_stats_s2_new_cond_feature.get('std', 1.0)
                if self.s2_new_cond_feature_std < 1e-6: self.s2_new_cond_feature_std = 1.0
                self.logger.info(f"Stage2Dataset (train): 已載入 Stage2 新條件 ({self.s2_new_cond_col_name}) 的正規化統計量: mean={self.s2_new_cond_feature_mean:.4f}, std={self.s2_new_cond_feature_std:.4f}")
            else: # 如果沒傳，則在 S3 train dataset 內部計算 (通常不建議，應由 S2 dataset 提供)
                self.logger.warning(f"Stage2Dataset (train): 未從 Stage2Dataset 接收 Stage2 新條件 ({self.s2_new_cond_col_name}) 的正規化統計量。將在內部計算。")
                self.norm_stats_s2_new_cond_feature = self._calculate_norm_stats(self.s2_new_cond_original_values_np, self.s2_new_cond_col_name, "S2 new cond")
                self.s2_new_cond_feature_mean = self.norm_stats_s2_new_cond_feature['mean']
                self.s2_new_cond_feature_std = self.norm_stats_s2_new_cond_feature['std']


            # S3 新條件特徵的正規化統計
            self.norm_stats_s3_new_cond_feature = self._calculate_norm_stats(self.s3_new_cond_original_values_np, self.s3_new_cond_col_name, "S3 new cond")
            self.s3_new_cond_feature_mean = self.norm_stats_s3_new_cond_feature['mean']
            self.s3_new_cond_feature_std = self.norm_stats_s3_new_cond_feature['std']
            self.logger.info(f"Stage2Dataset (train): 計算得到 Stage3 新條件 ({self.s3_new_cond_col_name}) 的正規化統計量: mean={self.s3_new_cond_feature_mean:.4f}, std={self.s3_new_cond_feature_std:.4f}")

            # S3 目標流量圖
            self.average_flow_map_dict_s3 = self._calculate_stage3_target_flows()
            if not self.average_flow_map_dict_s3: self.logger.warning("Stage3Dataset (train): _calculate_stage3_target_flows() 返回一個空字典。")
            
            # S3 目標流量的專用正規化統計
            if self.average_flow_map_dict_s3:
                all_s3_target_maps = np.array(list(self.average_flow_map_dict_s3.values()))
                if all_s3_target_maps.size > 0:
                    self.norm_stats_stage3_target = self._calculate_norm_stats(all_s3_target_maps.flatten(), "S3 Target", "S3 Target") # 使用 flatten 後的數據
                    self.logger.info(f"Stage2Dataset (train): 計算得到 Stage3 目標流量的專用正規化統計量: mean={self.norm_stats_stage3_target['mean']:.4f}, std={self.norm_stats_stage3_target['std']:.4f}")
                else:
                    self.logger.warning("Stage2Dataset (train): average_flow_map_dict_s3 中的值為空數組，無法計算 Stage3 目標專用統計量。使用預設值。")
                    self.norm_stats_stage3_target = {'mean': 0.0, 'std': 1.0}
            else:
                self.logger.warning("Stage2Dataset (train): average_flow_map_dict_s3 為空，無法計算 Stage3 目標專用統計量。使用預設值。")
                self.norm_stats_stage3_target = {'mean': 0.0, 'std': 1.0}

        elif self.mode == 'val' or self.mode == 'test':
            # 載入 S2 新條件的統計數據
            if s2_new_cond_feature_norm_stats_from_s2_train is None: raise ValueError(f"Stage3 {self.mode} mode 需要從 S2 訓練集傳入 s2_new_cond_feature_norm_stats。")
            self.norm_stats_s2_new_cond_feature = s2_new_cond_feature_norm_stats_from_s2_train
            self.s2_new_cond_feature_mean = self.norm_stats_s2_new_cond_feature.get('mean', 0.0)
            self.s2_new_cond_feature_std = self.norm_stats_s2_new_cond_feature.get('std', 1.0)
            if self.s2_new_cond_feature_std < 1e-6: self.s2_new_cond_feature_std = 1.0
            self.logger.info(f"Stage3Dataset ({self.mode}): 已載入 Stage2 新條件 ({self.s2_new_cond_col_name}) 的正規化統計量: mean={self.s2_new_cond_feature_mean:.4f}, std={self.s2_new_cond_feature_std:.4f}")

            # 載入 S3 新條件的統計數據
            if s3_new_cond_feature_norm_stats_from_train is None: raise ValueError(f"Stage3 {self.mode} mode 需要從訓練集傳入 s3_new_cond_feature_norm_stats。")
            self.norm_stats_s3_new_cond_feature = s3_new_cond_feature_norm_stats_from_train
            self.s3_new_cond_feature_mean = self.norm_stats_s3_new_cond_feature.get('mean', 0.0)
            self.s3_new_cond_feature_std = self.norm_stats_s3_new_cond_feature.get('std', 1.0)
            if self.s3_new_cond_feature_std < 1e-6: self.s3_new_cond_feature_std = 1.0
            self.logger.info(f"Stage3Dataset ({self.mode}): 已載入 Stage3 新條件 ({self.s3_new_cond_col_name}) 的正規化統計量: mean={self.s3_new_cond_feature_mean:.4f}, std={self.s3_new_cond_feature_std:.4f}")

            # 載入 S3 目標流量圖
            if stage3_avg_flow_map_dict_from_train is None: raise ValueError(f"Stage3 {self.mode} mode 需要從訓練集傳入 stage3_avg_flow_map_dict。")
            if not isinstance(stage3_avg_flow_map_dict_from_train, dict): raise TypeError(f"Stage3 {self.mode} mode: stage3_avg_flow_map_dict_from_train 必須是字典。")
            self.average_flow_map_dict_s3 = stage3_avg_flow_map_dict_from_train
            self.logger.info(f"Stage3Dataset ({self.mode}): 已載入 Stage3 平均流量圖字典 (包含 {len(self.average_flow_map_dict_s3)} 個條目)。")

            # 載入 S3 目標流量的專用正規化統計
            if stage3_target_norm_stats_from_train is None: raise ValueError(f"Stage3 {self.mode} mode 需要從訓練集傳入 stage3_target_norm_stats。")
            self.norm_stats_stage3_target = stage3_target_norm_stats_from_train
            s3_target_mean = self.norm_stats_stage3_target.get('mean', 0.0)
            s3_target_std = self.norm_stats_stage3_target.get('std', 1.0)
            if s3_target_std < 1e-6: s3_target_std = 1.0 # 再次檢查
            self.logger.info(f"Stage3Dataset ({self.mode}): 已載入 Stage3 目標流量的專用正規化統計量: mean={s3_target_mean:.4f}, std={s3_target_std:.4f}")
        else:
            raise ValueError(f"未知的 Stage3Dataset mode: {self.mode}")

        # Final checks for critical attributes
        for attr_name in ['average_flow_map_dict_s3', 'norm_stats_s3_new_cond_feature', 'norm_stats_stage3_target', 'norm_stats_s2_new_cond_feature']:
            if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
                self.logger.error(f"CRITICAL ERROR FINAL CHECK (mode={self.mode}): 屬性 '{attr_name}' 在 Stage3Dataset __init__ 結束時缺失或為 None!")
        self.logger.info(f"Stage3Dataset __init__ (mode={self.mode}) COMPLETED.")

    def _calculate_category_vector(self, values_np: np.ndarray, op: str, threshold: Any, col_name_for_log: str, cond_stage_log_prefix: str) -> np.ndarray:
        num_nan = np.isnan(values_np).sum()
        if num_nan > 0: self.logger.warning(f"Stage3Dataset ({cond_stage_log_prefix}, mode={self.mode}): 欄位 '{col_name_for_log}' 包含 {num_nan} 個 NaN。比較時 NaN 通常結果為 False。")
        
        series_vals = pd.Series(values_np)
        try:
            thresh_val = float(threshold)
            cat_0_desc, cat_1_desc = "", ""
            if op == "<=": condition_met_mask = (series_vals <= thresh_val); cat_0_desc=f"'{col_name_for_log}' <= {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' > {thresh_val}"
            elif op == ">": condition_met_mask = (series_vals > thresh_val); cat_0_desc=f"'{col_name_for_log}' > {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' <= {thresh_val}"
            elif op == "<": condition_met_mask = (series_vals < thresh_val); cat_0_desc=f"'{col_name_for_log}' < {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' >= {thresh_val}"
            elif op == ">=": condition_met_mask = (series_vals >= thresh_val); cat_0_desc=f"'{col_name_for_log}' >= {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' < {thresh_val}"
            elif op == "==": condition_met_mask = (series_vals == thresh_val); cat_0_desc=f"'{col_name_for_log}' == {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' != {thresh_val}"
            elif op == "!=": condition_met_mask = (series_vals != thresh_val); cat_0_desc=f"'{col_name_for_log}' != {thresh_val}"; cat_1_desc=f"'{col_name_for_log}' == {thresh_val}"
            else:
                self.logger.warning(f"Stage3Dataset ({cond_stage_log_prefix}, mode={self.mode}): 未明確處理運算符 '{op}' for column '{col_name_for_log}'，預設分類為 ({col_name_for_log} <= {thresh_val}) 為類別0。")
                condition_met_mask = (series_vals <= thresh_val); cat_0_desc=f"'{col_name_for_log}' <= {thresh_val} (預設)"; cat_1_desc=f"'{col_name_for_log}' > {thresh_val} (預設)"
            
            category_vector = (~condition_met_mask).astype(int) # 條件滿足為分支0 (主要分支), 不滿足為分支1
            self.logger.info(f"Stage3Dataset ({cond_stage_log_prefix}, mode={self.mode}): 條件 ('{col_name_for_log}') 分類邏輯 -> 類別0 (主要條件滿足): {cat_0_desc}; 類別1 (不滿足): {cat_1_desc}")
            unique_cats, counts_cats = np.unique(category_vector, return_counts=True)
            self.logger.info(f"  - 分類 ('{col_name_for_log}') 分佈: {dict(zip(unique_cats, counts_cats))}")
            return category_vector
        except ValueError:
            self.logger.error(f"Stage3Dataset ({cond_stage_log_prefix}, mode={self.mode}): 閾值 '{threshold}' for '{col_name_for_log}' 無法轉換為浮點數。所有樣本類別將設為0。")
            return np.zeros(len(self.df_s3), dtype=int)

    def _calculate_norm_stats(self, values_np: np.ndarray, col_name_for_log: str, cond_stage_log_prefix: str) -> Dict[str, float]:
        valid_values = values_np[~np.isnan(values_np)]
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
        else:
            self.logger.warning(f"Stage3Dataset ({cond_stage_log_prefix}, {self.mode}): 欄位 '{col_name_for_log}' 中沒有有效的數值用於計算正規化統計量。將使用 mean=0, std=1。")
            return {'mean': 0.0, 'std': 1.0}
        if std_val < 1e-6:
            self.logger.warning(f"Stage3Dataset ({cond_stage_log_prefix}, {self.mode}): 計算得到欄位 '{col_name_for_log}' 標準差 ({std_val:.4f}) 過小，將其設為 1.0。")
            std_val = 1.0
        return {'mean': mean_val, 'std': std_val}

    def __len__(self) -> int:
        return len(self.df_s3)

    def _calculate_stage3_target_flows(self) -> Dict[Tuple[int, int, int, int], np.ndarray]:
        self.logger.info(f"Stage3Dataset (mode={self.mode}): 計算複合條件 (小時類別, 假日, S2條件類別, S3條件類別) 的目標平均流量...")
        avg_flows_s3: Dict[Tuple[int, int, int, int], np.ndarray] = {}

        if not hasattr(self, 'sorted_flow_columns') or not self.sorted_flow_columns: return {}
        missing_cols = [col for col in self.sorted_flow_columns if col not in self.df_s3.columns]
        if missing_cols: self.logger.error(f"... _calc_s3_targets: 流量欄位缺失: {missing_cols}"); return {}

        flow_data_for_calc_s3 = self.df_s3[self.sorted_flow_columns].values.astype(np.float32)
        
        grouping_df_s3 = pd.DataFrame({
            'hour_category': self.hour_category_for_target_grouping_np_s3,
            'is_holiday': self.is_holiday_for_target_np_s3,
            's2_cond_category': self.s2_new_cond_category_for_target_np,
            's3_cond_category': self.s3_new_cond_category_for_target_np
        })
        if grouping_df_s3.empty: return {}

        try:
            grouped_s3 = grouping_df_s3.groupby(['hour_category', 'is_holiday', 's2_cond_category', 's3_cond_category'], observed=False)
        except Exception as e: self.logger.error(f"... _calc_s3_targets: Groupby 錯誤: {e}"); return {}
        if not grouped_s3.groups or all(idx.empty for idx in grouped_s3.groups.values()): return {}

        self.logger.info(f"Stage3 Target Calculation (mode={self.mode}): 樣本數分佈 (hr_cat, hol, s2_cat, s3_cat): count")
        for group_key_s3, group_indices_s3 in grouped_s3.indices.items():
            if len(group_indices_s3) == 0: continue
            hr_cat, is_hol, s2_cat, s3_cat = group_key_s3
            count = len(group_indices_s3)
            self.logger.info(f"  - ({hr_cat}, {is_hol}, {s2_cat}, {s3_cat}): {count} samples")

            group_flows = flow_data_for_calc_s3[group_indices_s3]
            mean_flow_flat = np.nanmean(group_flows, axis=0)
            mean_flow_flat[np.isnan(mean_flow_flat)] = 0
            avg_flows_s3[(hr_cat, int(is_hol), int(s2_cat), int(s3_cat))] = mean_flow_flat.reshape(self.H, self.W)
        
        if not avg_flows_s3: self.logger.warning(f"Stage3Dataset (mode={self.mode}): _calculate_stage3_target_flows - avg_flows_s3 字典為空。")
        self.logger.info(f"Stage3Dataset (mode={self.mode}): 計算完成 {len(avg_flows_s3)} 個 Stage3 條件的目標平均流量圖。")
        return avg_flows_s3

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- Stage3 模型的主要輸入 ---
        # Condition 1 for S3 model: Stage2 model output (正規化)
        s2_model_output_grid_sample_norm = self.stage2_model_outputs_np[idx] # 預期是 (1, D, H, W)
        if s2_model_output_grid_sample_norm.shape[0] != 1: # 確保通道數為1
            self.logger.warning(f"__getitem__ (idx {idx}): S2 model output (cond1 for S3) has {s2_model_output_grid_sample_norm.shape[0]} channels, expected 1. Using first.")
            s2_model_output_grid_sample_norm = s2_model_output_grid_sample_norm[0:1, ...]
        condition1_s3_tensor_norm = torch.from_numpy(s2_model_output_grid_sample_norm.astype(np.float32))

        # Condition 2 for S3 model: Stage3 new feature grid (正規化)
        original_s3_cond_value = self.s3_new_cond_original_values_np[idx]
        current_s3_cond_std = self.norm_stats_s3_new_cond_feature['std'] if self.norm_stats_s3_new_cond_feature['std'] > 1e-6 else 1.0
        normalized_s3_cond_value = (original_s3_cond_value - self.norm_stats_s3_new_cond_feature['mean']) / current_s3_cond_std \
            if not np.isnan(original_s3_cond_value) else 0.0
        condition2_s3_tensor_norm = torch.full(
            (1, self.D, self.H, self.W), float(normalized_s3_cond_value), dtype=torch.float32
        )

        # Target for S3 model (正規化)
        hr_cat_s3 = self.hour_category_for_target_grouping_np_s3[idx]
        is_hol_s3 = self.is_holiday_for_target_np_s3[idx]
        s2_cond_cat_s3 = self.s2_new_cond_category_for_target_np[idx]
        s3_cond_cat_s3 = self.s3_new_cond_category_for_target_np[idx]
        target_key_s3 = (hr_cat_s3, is_hol_s3, s2_cond_cat_s3, s3_cond_cat_s3)
        
        target_avg_flow_s3_np = self.average_flow_map_dict_s3.get(target_key_s3)
        if target_avg_flow_s3_np is None:
            self.logger.debug(f"Stage3Dataset (idx {idx}, mode={self.mode}): 未找到 S3 目標鍵 {target_key_s3}，使用零值網格。可用鍵: {len(self.average_flow_map_dict_s3)}")
            target_avg_flow_s3_np = np.zeros((self.H, self.W), dtype=np.float32)
        
        # 決定目標流量圖正規化時使用的均值和標準差
        target_mean_to_use_for_norm: float
        target_std_to_use_for_norm: float

        if hasattr(self, 'norm_stats_stage3_target') and self.norm_stats_stage3_target is not None:
            target_mean_to_use_for_norm = self.norm_stats_stage3_target['mean']
            target_std_to_use_for_norm = self.norm_stats_stage3_target['std']
            # self.logger.debug(f"Stage2Dataset __getitem__: Using dedicated S2 target norm: mean={target_mean_to_use_for_norm}, std={target_std_to_use_for_norm}")
        else: # 回退
            target_mean_to_use_for_norm = self.config.get("cached_basemodel_mean", 0.0)
            target_std_to_use_for_norm = self.config.get("cached_basemodel_std", 1.0)
            # self.logger.debug(f"Stage2Dataset __getitem__: Using cached_basemodel_stats for S2 target norm: mean={target_mean_to_use_for_norm}, std={target_std_to_use_for_norm}")
        
        if target_std_to_use_for_norm < 1e-6: target_std_to_use_for_norm =  1.0


        norm_target_s3_np = (target_avg_flow_s3_np - target_mean_to_use_for_norm) / target_std_to_use_for_norm
        
        target_flow_tensor = torch.from_numpy(norm_target_s3_np).float().reshape(
            self.image_channels_target, self.D, self.H, self.W
        )

        # --- 額外資訊，主要用於評估時重建 Basemodel 和 Stage2 模型的輸入 ---
        original_hour_scalar_tensor = torch.tensor(self.hours_for_target_np_s3[idx], dtype=torch.long)
        original_is_holiday_scalar_tensor = torch.tensor(is_hol_s3, dtype=torch.long) # is_hol_s3 已經是 int

        # Basemodel output grid (正規化) - 作為 Stage2 評估時的條件1
        bm_output_grid_sample_norm = self.basemodel_outputs_np[idx] # 預期是 (1, D, H, W)
        if bm_output_grid_sample_norm.shape[0] != 1:
             bm_output_grid_sample_norm = bm_output_grid_sample_norm[0:1, ...] # 取第一個通道
        bm_output_grid_for_s2eval_tensor_norm = torch.from_numpy(bm_output_grid_sample_norm.astype(np.float32))
        
        # Stage2 new feature grid (正規化) - 作為 Stage2 評估時的條件2
        original_s2_cond_value = self.s2_new_cond_original_values_np[idx]
        current_s2_cond_std = self.norm_stats_s2_new_cond_feature['std'] if self.norm_stats_s2_new_cond_feature['std'] > 1e-6 else 1.0
        normalized_s2_cond_value = (original_s2_cond_value - self.norm_stats_s2_new_cond_feature['mean']) / current_s2_cond_std \
            if not np.isnan(original_s2_cond_value) else 0.0
        s2_new_feature_grid_for_s2eval_tensor_norm = torch.full(
            (1, self.D, self.H, self.W), float(normalized_s2_cond_value), dtype=torch.float32
        )
        
        # Stage2 原始特徵值 (純量) - 用於可能的日誌或調試
        s2_original_feature_scalar_tensor = torch.tensor(original_s2_cond_value if not np.isnan(original_s2_cond_value) else 0.0, dtype=torch.float32)
        # Stage3 原始特徵值 (純量) - 用於可能的日誌或調試
        s3_original_feature_scalar_tensor = torch.tensor(original_s3_cond_value if not np.isnan(original_s3_cond_value) else 0.0, dtype=torch.float32)


        return (target_s3_tensor_norm,
                condition1_s3_tensor_norm,  # S2 model output (norm)
                condition2_s3_tensor_norm,  # S3 new feature grid (norm)
                original_hour_scalar_tensor,
                original_is_holiday_scalar_tensor,
                bm_output_grid_for_s2eval_tensor_norm, # BM output (norm)
                s2_new_feature_grid_for_s2eval_tensor_norm, # S2 new feature grid (norm)
                s2_original_feature_scalar_tensor, # S2 原始特徵純量
                s3_original_feature_scalar_tensor  # S3 原始特徵純量
               )

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
                        prefix: str = "test_eval" # 檔名前綴
                       ):
    """
    視覺化預測結果與真實值的比較 (針對 DDPM_Long-term.ipynb 的數據結構)。
    包含生成結果、真實數據、以及誤差（MSE、MAE、MAPE、SMAPE）的網格熱力圖。
    """
    save_dir = config.get("stage2_model_save_dir", config["save_dir"])
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

    epsilon = 1e-8 # 避免除以零
    mse_matrix = (gen_data_to_plot - orig_data_to_plot) ** 2
    mae_matrix = np.abs(gen_data_to_plot - orig_data_to_plot)
    mape_matrix = np.abs((orig_data_to_plot - gen_data_to_plot) / (np.abs(orig_data_to_plot) + epsilon)) * 100
    smape_matrix = np.abs(gen_data_to_plot - orig_data_to_plot) / ((np.abs(orig_data_to_plot) + np.abs(gen_data_to_plot))/2 + epsilon) * 100 



    overall_mse = np.mean(mse_matrix)
    overall_mae = np.mean(mae_matrix)
    overall_mape = np.mean(mape_matrix[np.isfinite(mape_matrix)])
    overall_smape = np.mean(smape_matrix[np.isfinite(smape_matrix)])


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
                        prefix: str = "test_eval"
                       ):
    logger_func = logging.getLogger(__name__) # 確保 logger 在函數作用域內可用
    save_dir = config.get("stage2_model_save_dir", config["save_dir"]) # 優先使用 stage2 的儲存目錄
    os.makedirs(save_dir, exist_ok=True)

    H, W = config["H"], config["W"]

    # 從 config 中獲取網格映射信息，這些資訊應在加載 Basemodel 檢查點時被緩存
    sorted_flow_columns_map = config.get("cached_basemodel_sorted_flow_columns")
    grid_idx_to_rc_map_plot = config.get("cached_basemodel_grid_idx_to_rc_map")
    selected_sensor_info_plot = config.get("cached_basemodel_selected_sensor_info")

    if not all([sorted_flow_columns_map, grid_idx_to_rc_map_plot, selected_sensor_info_plot]):
        logger_func.error("plot_grid_with_error_long_term: CONFIG 中缺少必要的網格映射資訊 "
                          "(cached_basemodel_sorted_flow_columns, cached_basemodel_grid_idx_to_rc_map, "
                          "cached_basemodel_selected_sensor_info)。請確保 Basemodel 檢查點已載入這些資訊到 CONFIG。")
        return

    selected_sensor_info_dict = {info['name']: (info['lon'], info['lat'])
                                 for info in selected_sensor_info_plot if isinstance(info, dict) and 'name' in info}

    actual_sensor_lons = []
    actual_sensor_lats = []
    valid_grid_indices_flat = []

    for flat_grid_idx in range(H * W):
        if flat_grid_idx < len(sorted_flow_columns_map):
            col_name = sorted_flow_columns_map[flat_grid_idx]
            if col_name in selected_sensor_info_dict:
                lon, lat = selected_sensor_info_dict[col_name]
                actual_sensor_lons.append(lon)
                actual_sensor_lats.append(lat)
                valid_grid_indices_flat.append(flat_grid_idx)
            else:
                logger_func.debug(f"plot_grid_with_error: Column {col_name} (grid_idx {flat_grid_idx}) not in selected_sensor_info_dict.")
        else:
            logger_func.warning(f"plot_grid_with_error: flat_grid_idx {flat_grid_idx} out of bounds for sorted_flow_columns (len: {len(sorted_flow_columns_map)}).")

    if not actual_sensor_lons:
        logger_func.error("plot_grid_with_error: Could not retrieve coordinates for any grid points.")
        return

    cdict_red_to_black = {
        'red':   ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    red_to_black_cmap = mcolors.LinearSegmentedColormap('RedToBlack', cdict_red_to_black)

    for metric_name, error_grid_flat in error_metrics_grids.items():
        if not isinstance(error_grid_flat, np.ndarray) or error_grid_flat.ndim == 0 or error_grid_flat.shape[0] != H*W :
            logger_func.error(f"Dimension of error_grid for metric {metric_name} ({error_grid_flat.shape if isinstance(error_grid_flat, np.ndarray) else type(error_grid_flat)}) is incorrect. Expected ({H*W},). Skipping plot.")
            continue

        # 過濾掉 error_grid_flat 中對應 valid_grid_indices_flat 的值
        # 確保只使用有效的網格點對應的誤差值
        if len(valid_grid_indices_flat) != len(actual_sensor_lons):
             logger_func.warning("Mismatch between valid_grid_indices_flat and actual_sensor_lons. Plot may be incorrect.")
             # Fallback or error, for now, we proceed but this indicates an issue.

        # 僅提取有效網格點的誤差值進行繪圖
        error_values_for_plot = error_grid_flat[valid_grid_indices_flat]
        
        if len(error_values_for_plot) == 0:
            logger_func.warning(f"No valid error values to plot for metric {metric_name} after filtering. Skipping plot.")
            continue
        
        # 處理可能的 inf/nan 值，避免繪圖錯誤
        error_values_for_plot_finite = error_values_for_plot[np.isfinite(error_values_for_plot)]
        if len(error_values_for_plot_finite) == 0 and len(error_values_for_plot) > 0 : # 如果全是inf/nan
             logger_func.warning(f"All error values for metric {metric_name} are non-finite. Setting to 0 for plotting.")
             error_values_for_plot_display = np.full_like(error_values_for_plot, np.nan, dtype=float)
             min_val, max_val = 0,1 # 給定預設範圍
        elif len(error_values_for_plot_finite) == 0 and len(error_values_for_plot) == 0:
            logger_func.warning(f"No error values to plot for metric {metric_name}.")
            continue
        else:
            error_values_for_plot_display = np.where(np.isfinite(error_values_for_plot), error_values_for_plot, np.nan) # 非有限值設為nan
            min_val = np.nanmin(error_values_for_plot_display)
            max_val = np.nanmax(error_values_for_plot_display)


        plt.figure(figsize=(12, 12))
        # 使用新的 red_to_black_cmap，並傳遞 vmin 和 vmax
        if min_val == max_val: # 如果所有有限值都相同
            if np.isnan(min_val): # 如果全是nan
                 min_val, max_val = 0,1
            else: # 如果是同一個數
                 min_val = min_val - 0.5 if min_val != 0 else -0.5
                 max_val = max_val + 0.5 if max_val != 0 else 0.5
        
        scatter = plt.scatter(actual_sensor_lons, actual_sensor_lats, c=error_values_for_plot_display, 
                                cmap=red_to_black_cmap, marker='s', s=100, vmin=min_val, vmax=max_val)
        
        plt.colorbar(scatter, label=metric_name)

        if metric_name.upper() != 'MSE': # 只有非 MSE 的指標圖才在網格上顯示數字
            for i in range(len(actual_sensor_lons)):
                val_to_text = error_values_for_plot_display[i]
                if not np.isnan(val_to_text): # 只顯示有限的數值
                    plt.text(actual_sensor_lons[i], actual_sensor_lats[i],
                             f'{val_to_text:.0f}',
                             fontsize=6, color='white', ha='center', va='center')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Geographic Grid Error Heatmap - {metric_name.upper()} ({prefix})")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box') # 確保地理比例正確
        plt.savefig(os.path.join(save_dir, f'{prefix}_grid_{metric_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger_func.info(f"Saved {metric_name} geographic grid error map for {prefix}.")
                        

def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

@torch.no_grad()
def evaluate_stage2_models(
    stage2_model_trained: 'DDPM3D',
    basemodel_eval_instance: 'DDPM3D',
    dataloader_s2: DataLoader,
    inception_model_fid: nn.Module,
    config: Dict[str, Any],
    max_samples_for_fid: Optional[int] = None,
    prefix: str = "stage2_eval"
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]: 
    logger.info(f"===== 開始 Stage2 模型評估 (比較 {prefix}) =====")
    stage2_model_trained.eval()
    basemodel_eval_instance.eval()
    inception_model_fid.eval()

    # 從 CONFIG 或 dataset 物件獲取 Stage2 目標的反正規化統計量
    # 注意: dataloader_s2.dataset 應該是 Stage2Dataset 的實例
    dataset_s2_obj_for_eval = dataloader_s2.dataset
    
    target_mean_for_denorm: float
    target_std_for_denorm: float

    if hasattr(dataset_s2_obj_for_eval, 'norm_stats_stage2_target') and \
       dataset_s2_obj_for_eval.norm_stats_stage2_target is not None:
        current_s2_target_stats = dataset_s2_obj_for_eval.norm_stats_stage2_target
        target_mean_for_denorm = current_s2_target_stats['mean']
        target_std_for_denorm = current_s2_target_stats['std']
        logger.info(f"evaluate_stage2_models: 使用 Stage2Dataset 提供的目標專用統計量進行反正規化: mean={target_mean_for_denorm:.4f}, std={target_std_for_denorm:.4f}")
    else: # 回退到 cached_basemodel_stats (這應該是個備用方案，正常情況下 Stage2Dataset 會提供)
        target_mean_for_denorm = config.get("cached_basemodel_mean")
        target_std_for_denorm = config.get("cached_basemodel_std")
        logger.warning(f"evaluate_stage2_models: Stage2Dataset 未提供 norm_stats_stage2_target。回退使用 cached_basemodel_stats: mean={target_mean_for_denorm}, std={target_std_for_denorm}")

    if target_mean_for_denorm is None or target_std_for_denorm is None:
        logger.error(f"evaluate_stage2_models: 無法獲取用於反正規化的均值或標準差。")
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape": float('nan'), "smape": float('nan'), "fid": float('nan')}
        nan_grids_dict = {'MSE': np.array([np.nan]), 'MAE': np.array([np.nan]), 'MAPE': np.array([np.nan]), 'SMAPE': np.array([np.nan])} # 提供一個預設的 ndarray
        return {"stage2_model": nan_metrics, "basemodel_on_s2_data": nan_metrics}, \
               {"stage2_model": nan_grids_dict, "basemodel_on_s2_data": nan_grids_dict}
    
    if target_std_for_denorm < 1e-6: # 已在前面處理過，但再次檢查以防萬一
        logger.warning(f"evaluate_stage2_models: 用於反正規化的標準差 ({target_std_for_denorm}) 過小，將其視為 1.0。")
        target_std_for_denorm = 1.0
    max_fid_samples_actual = len(dataset_s2_obj_for_eval)


    if max_samples_for_fid is not None:
        max_fid_samples_actual = min(max_samples_for_fid, max_fid_samples_actual)
    logger.info(f"將為 FID 計算收集最多 {max_fid_samples_actual} 個樣本。")


    pbar_s2_eval = tqdm(dataloader_s2, desc=f"Stage2 評估 ({prefix})", leave=False)
    for target_s2_eval_norm, bm_out_eval_cond, uv_eval_cond, \
        orig_hr_eval_s, orig_is_hol_eval_s in pbar_s2_eval:

        target_s2_eval_norm = target_s2_eval_norm.to(config["device"])
        bm_out_eval_cond = bm_out_eval_cond.to(config["device"])
        uv_eval_cond = uv_eval_cond.to(config["device"])
        orig_hr_eval_s = orig_hr_eval_s.to(config["device"])
        orig_is_hol_eval_s = orig_is_hol_eval_s.to(config["device"])

        current_batch_size = target_s2_eval_norm.shape[0]

        # 1. Stage2 模型生成
        s2_generated_eval_norm = stage2_model_trained.sample(
            batch_size=current_batch_size,
            basemodel_output_grid_batch=bm_out_eval_cond,
            new_condition_feature_grid_batch=uv_eval_cond
        )
        s2_generated_eval_denorm = s2_generated_eval_norm * target_std_for_denorm + target_mean_for_denorm

        # 2. Basemodel 在相同原始條件下的生成
        bm_generated_norm_on_s2_conditions = basemodel_eval_instance.sample(
            batch_size=current_batch_size,
            hour_scalars_batch=orig_hr_eval_s,
            is_holiday_scalars_batch=orig_is_hol_eval_s
        )
        bm_generated_denorm_on_s2_conditions = bm_generated_norm_on_s2_conditions * target_std_for_denorm + target_mean_for_denorm

        # Stage2 的目標反正規化
        s2_target_eval_denorm = target_s2_eval_norm * target_std_for_denorm + target_mean_for_denorm

        s2_generated_eval_denorm = torch.clamp(s2_generated_eval_denorm, min=0.0)
        bm_generated_denorm_on_s2_conditions = torch.clamp(bm_generated_denorm_on_s2_conditions, min=0.0)

        # 為 FID 收集正規化樣本
        samples_collected_so_far = sum(s.shape[0] for s in all_s2_target_norm_for_fid_list) # 以目標列表長度為準
        if samples_collected_so_far < max_fid_samples_actual:
            remaining_needed_fid = max_fid_samples_actual - samples_collected_so_far
            samples_to_add_fid = min(current_batch_size, remaining_needed_fid)
            if samples_to_add_fid > 0:
                all_s2_generated_norm_for_fid_list.append(s2_generated_eval_norm[:samples_to_add_fid].cpu())
                all_s2_target_norm_for_fid_list.append(target_s2_eval_norm[:samples_to_add_fid].cpu())
                all_bm_generated_norm_for_fid_on_s2_data_list.append(bm_generated_norm_on_s2_conditions[:samples_to_add_fid].cpu())

    if not all_s2_target_denorm_list: # 如果沒有處理任何數據
        logger.warning(f"Stage2 評估 ({prefix}): 無數據處理或收集到。")
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape": float('nan'), "smape": float('nan'), "fid": float('nan')}
        nan_grids_dict = {'MSE': np.array([np.nan]), 'MAE': np.array([np.nan]), 'MAPE': np.array([np.nan]), 'SMAPE': np.array([np.nan])} # 提供一個預設的 ndarray
        return {"stage2_model": nan_metrics, "basemodel_on_s2_data": nan_metrics}, \
               {"stage2_model": nan_grids_dict, "basemodel_on_s2_data": nan_grids_dict}

    s2_target_all_t = torch.cat(all_s2_target_denorm_list, dim=0)
    s2_generated_all_t = torch.cat(all_s2_generated_denorm_list, dim=0)
    bm_generated_all_on_s2_data_t = torch.cat(all_bm_generated_denorm_on_s2_data_list, dim=0)

    logger.info(f"s2_target_all_t (反正規化後的Stage2目標) shape: {s2_target_all_t.shape}")
    logger.info(f"  min: {torch.min(s2_target_all_t).item():.2f}, max: {torch.max(s2_target_all_t).item():.2f}, mean: {torch.mean(s2_target_all_t).item():.2f}")
    logger.info(f"s2_generated_all_t (反正規化後的Stage2預測) shape: {s2_generated_all_t.shape}")
    logger.info(f"  min: {torch.min(s2_generated_all_t).item():.2f}, max: {torch.max(s2_generated_all_t).item():.2f}, mean: {torch.mean(s2_generated_all_t).item():.2f}")
    logger.info(f"bm_generated_all_on_s2_data_t (反正規化後的Basemodel預測) shape: {bm_generated_all_on_s2_data_t.shape}")
    logger.info(f"  min: {torch.min(bm_generated_all_on_s2_data_t).item():.2f}, max: {torch.max(bm_generated_all_on_s2_data_t).item():.2f}, mean: {torch.mean(bm_generated_all_on_s2_data_t).item():.2f}")

    epsilon = 1e-8
    results = {}
    error_grids_all_models: Dict[str, Dict[str, Any]] = {} # 值可以是 np.ndarray

    for model_name, pred_t in [("stage2_model", s2_generated_all_t),
                               ("basemodel_on_s2_data", bm_generated_all_on_s2_data_t)]:
        mse = F.mse_loss(pred_t, s2_target_all_t).item()
        mae = F.l1_loss(pred_t, s2_target_all_t).item()

        mape_tensor = torch.abs((s2_target_all_t - pred_t) / (torch.abs(s2_target_all_t) + epsilon)) * 100
        mape = torch.mean(mape_tensor[torch.isfinite(mape_tensor)]).item() if torch.isfinite(mape_tensor).any() else float('inf')

        smape_n = torch.abs(pred_t - s2_target_all_t)
        smape_d = (torch.abs(s2_target_all_t) + torch.abs(pred_t)) / 2.0 + epsilon
        smape_tensor = (smape_n / smape_d) * 100
        smape = torch.mean(smape_tensor[torch.isfinite(smape_tensor)]).item() if torch.isfinite(smape_tensor).any() else float('inf')

        fid = float('nan')
        current_generated_norm_for_fid_list_to_use = all_s2_generated_norm_for_fid_list

        if current_generated_norm_for_fid_list_to_use and all_s2_target_norm_for_fid_list:
            # 確保 FID 樣本列表不為空
            if not all(len(lst) > 0 for lst in [current_generated_norm_for_fid_list_to_use, all_s2_target_norm_for_fid_list]):
                 logger.warning(f"FID for {model_name}: Not enough batches collected for FID sample lists.")
            else:
                gen_fid_tensor = torch.cat(current_generated_norm_for_fid_list_to_use, dim=0)
                real_fid_tensor = torch.cat(all_s2_target_norm_for_fid_list, dim=0)
                
                # 截取到 max_fid_samples_actual
                gen_fid_tensor = gen_fid_tensor[:max_fid_samples_actual]
                real_fid_tensor = real_fid_tensor[:max_fid_samples_actual]

                num_fid = min(gen_fid_tensor.shape[0], real_fid_tensor.shape[0])
                if num_fid > 1: # FID 計算至少需要2個樣本才能計算協方差
                    logger.info(f"Calculating FID for {model_name} (vs S2 target) on {num_fid} samples...")
                    try:
                        act_gen = get_activations(gen_fid_tensor, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                        act_real = get_activations(real_fid_tensor, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                        if act_gen.shape[0] > 1 and act_real.shape[0] > 1: # 確保激活特徵也足夠
                            fid = calculate_fid(act_real, act_gen)
                        else:
                            logger.warning(f"FID for {model_name}: Insufficient features obtained after activations ({act_gen.shape[0]}, {act_real.shape[0]}).")
                    except Exception as e_fid:
                        logger.error(f"FID calculation for {model_name} failed: {e_fid}")
                else:
                    logger.warning(f"FID for {model_name}: Insufficient samples after concatenation and truncation ({num_fid}).")
        else:
            logger.warning(f"FID for {model_name}: FID sample lists were empty before concatenation.")

        results[model_name] = {"mse": mse, "mae": mae, "mape": mape, "smape": smape, "fid": fid if np.isfinite(fid) else float('nan')}
        logger.info(f"Metrics for {model_name} ({prefix}): {results[model_name]}")

        # 計算每個網格的誤差指標
        if pred_t.ndim == 5 and pred_t.shape[1] == config["image_channels"] and pred_t.shape[2:] == (config.get("D",1), config["H"], config["W"]):
            # 假設 C=1, D=1 for 2D grid error maps
            pred_squeezed_for_grid_error = pred_t.squeeze(1).squeeze(1) # (N, H, W)
            target_squeezed_for_grid_error = s2_target_all_t.squeeze(1).squeeze(1) # (N, H, W)

            mse_g_grid = torch.mean((pred_squeezed_for_grid_error - target_squeezed_for_grid_error)**2, dim=0).cpu().numpy() # (H,W)
            mae_g_grid = torch.mean(torch.abs(pred_squeezed_for_grid_error - target_squeezed_for_grid_error), dim=0).cpu().numpy()

            mape_g_t_grid = torch.abs((target_squeezed_for_grid_error - pred_squeezed_for_grid_error) / (torch.abs(target_squeezed_for_grid_error) + epsilon)) * 100
            mape_g_grid = torch.mean(mape_g_t_grid, dim=0).cpu().numpy()

            smape_n_g_grid = torch.abs(pred_squeezed_for_grid_error - target_squeezed_for_grid_error)
            smape_d_g_grid = (torch.abs(target_squeezed_for_grid_error) + torch.abs(pred_squeezed_for_grid_error))/2.0 + epsilon
            smape_g_t_grid = (smape_n_g_grid / smape_d_g_grid) * 100
            smape_g_grid = torch.mean(smape_g_t_grid, dim=0).cpu().numpy()

            error_grids_all_models[model_name] = {
                'MSE': mse_g_grid.flatten(), # 展平以便與 plot_grid_with_error_long_term 兼容
                'MAE': mae_g_grid.flatten(),
                'MAPE': mape_g_grid.flatten(),
                'SMAPE': smape_g_grid.flatten()
            }
        else:
            logger.warning(f"Prediction tensor shape mismatch for per-grid metrics ({model_name}). Pred shape: {pred_t.shape}. Skipping grid error calculation.")
            error_grids_all_models[model_name] = {m: np.full((config["H"] * config["W"],), np.nan) for m in ['MSE','MAE','MAPE','SMAPE']}


    logger.info(f"Generating visualizations for Stage2 evaluation ({prefix})...")
    
    # --- 視覺化 Stage2 Model vs Target ---
    # 繪製第一個樣本的比較
    if s2_target_all_t.shape[0] > 0: # 確保有樣本可繪製
        logger.info(f"  Visualizing Stage2 Model vs Target (sample 0)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=s2_generated_all_t[0:1].clone().cpu(), # 取第一個樣本
            original_all_denorm_t=s2_target_all_t[0:1].clone().cpu(),   # 對應的目標
            config=config,
            sample_idx_to_plot=0, # 傳遞給 visualize_predictions_long_term 的 sample_idx，因為它內部可能還會用
            prefix=f"{prefix}_S2_vs_Target_sample0" # 清晰的檔名前綴
        )
        
        # 繪製所有樣本平均後的比較圖
        logger.info(f"  Visualizing Stage2 Model vs Target (average)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=torch.mean(s2_generated_all_t, dim=0, keepdim=True).clone().cpu(), # 平均預測
            original_all_denorm_t=torch.mean(s2_target_all_t, dim=0, keepdim=True).clone().cpu(),     # 平均目標
            config=config,
            sample_idx_to_plot=None, # 指示 visualize_predictions_long_term 這是平均圖
            prefix=f"{prefix}_S2_vs_Target_avg"
        )

    # --- 視覺化 Base Model vs Target ---
    if bm_generated_all_on_s2_data_t.shape[0] > 0 and s2_target_all_t.shape[0] > 0: # 確保有樣本可繪製
        logger.info(f"  Visualizing Base Model vs Target (sample 0)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=bm_generated_all_on_s2_data_t[0:1].clone().cpu(), # Base Model 的第一個樣本預測
            original_all_denorm_t=s2_target_all_t[0:1].clone().cpu(),                # 對應的目標
            config=config,
            sample_idx_to_plot=0,
            prefix=f"{prefix}_BM_vs_Target_sample0" # 清晰的檔名前綴
        )

        # 繪製所有樣本平均後的比較圖
        logger.info(f"  Visualizing Base Model vs Target (average)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=torch.mean(bm_generated_all_on_s2_data_t, dim=0, keepdim=True).clone().cpu(), # Base Model 平均預測
            original_all_denorm_t=torch.mean(s2_target_all_t, dim=0, keepdim=True).clone().cpu(),     # 平均目標
            config=config,
            sample_idx_to_plot=None,
            prefix=f"{prefix}_BM_vs_Target_avg"
        )

    # --- 視覺化 Stage2 Model vs S3 Target ---
    if s2_model_generated_all_on_s3_data_t.shape[0] > 0 and s3_target_all_t.shape[0] > 0:
        logger.info(f"  Visualizing Stage2 Model vs S3 Target (sample 0)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=s2_model_generated_all_on_s3_data_t[0:1].clone().cpu(),
            original_all_denorm_t=s3_target_all_t[0:1].clone().cpu(),
            config=config,
            sample_idx_to_plot=0,
            prefix=f"{prefix}_S2_vs_S3Target_sample0"
        )
        logger.info(f"  Visualizing Stage2 Model vs S3 Target (average)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=torch.mean(s2_model_generated_all_on_s3_data_t, dim=0, keepdim=True).clone().cpu(),
            original_all_denorm_t=torch.mean(s3_target_all_t, dim=0, keepdim=True).clone().cpu(),
            config=config,
            sample_idx_to_plot=None,
            prefix=f"{prefix}_S2_vs_S3Target_avg"
        )

    # --- 視覺化 Base Model vs S3 Target ---
    if bm_generated_all_on_s3_data_t.shape[0] > 0 and s3_target_all_t.shape[0] > 0: 
        logger.info(f"  Visualizing Base Model vs S3 Target (sample 0)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=bm_generated_all_on_s3_data_t[0:1].clone().cpu(), 
            original_all_denorm_t=s3_target_all_t[0:1].clone().cpu(),                
            config=config,
            sample_idx_to_plot=0,
            prefix=f"{prefix}_BM_vs_S3Target_sample0" 
        )
        logger.info(f"  Visualizing Base Model vs S3 Target (average)...")
        visualize_predictions_long_term(
            generated_all_denorm_t=torch.mean(bm_generated_all_on_s3_data_t, dim=0, keepdim=True).clone().cpu(), 
            original_all_denorm_t=torch.mean(s3_target_all_t, dim=0, keepdim=True).clone().cpu(),     
            config=config,
            sample_idx_to_plot=None,
            prefix=f"{prefix}_BM_vs_S3Target_avg"
        )

    # 繪製誤差地理圖
    if "stage3_model" in error_grids_all_models and isinstance(error_grids_all_models["stage3_model"], dict):
        plot_grid_with_error_long_term(
            dataset_s3_obj_for_eval, 
            error_grids_all_models["stage3_model"],
            config,
            f"{prefix}_stage3_model" # Clarified prefix
        )
    if "stage2_model_on_s3_data" in error_grids_all_models and isinstance(error_grids_all_models["stage2_model_on_s3_data"], dict): # Added for S2 model
        plot_grid_with_error_long_term(
            dataset_s3_obj_for_eval,
            error_grids_all_models["stage2_model_on_s3_data"],
            config,
            f"{prefix}_stage2_model" # Clarified prefix
        )
    if "basemodel_on_s3_data" in error_grids_all_models and isinstance(error_grids_all_models["basemodel_on_s3_data"], dict):
        plot_grid_with_error_long_term(
            dataset_s3_obj_for_eval, 
            error_grids_all_models["basemodel_on_s3_data"],
            config,
            f"{prefix}_basemodel" 
        )

    # 計算並繪製誤差差異圖
    s3_model_errors = error_grids_all_models.get("stage3_model")
    s2_model_errors_on_s3 = error_grids_all_models.get("stage2_model_on_s3_data")
    bm_errors_on_s3 = error_grids_all_models.get("basemodel_on_s3_data")

    # S3 vs BM
    if isinstance(s3_model_errors, dict) and isinstance(bm_errors_on_s3, dict):
        error_metrics_difference_s3_bm = {}
        for metric_key in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
            if metric_key in s3_model_errors and isinstance(s3_model_errors[metric_key], np.ndarray) and \
               metric_key in bm_errors_on_s3 and isinstance(bm_errors_on_s3[metric_key], np.ndarray) and \
               s3_model_errors[metric_key].shape == bm_errors_on_s3[metric_key].shape:
                difference_grid = s3_model_errors[metric_key] - bm_errors_on_s3[metric_key]
                error_metrics_difference_s3_bm[f"Diff_{metric_key}_(S3-BM)"] = difference_grid
            else:
                logger.warning(f"無法計算指標 '{metric_key}' 的差異網格 (S3-BM)，因數據缺失、類型錯誤或形狀不匹配。")
        if error_metrics_difference_s3_bm:
            plot_grid_with_error_long_term(
                dataset_s3_obj_for_eval,
                error_metrics_difference_s3_bm,
                config,
                f"{prefix}_diff_S3_minus_BM"
            )
    
    # S3 vs S2
    if isinstance(s3_model_errors, dict) and isinstance(s2_model_errors_on_s3, dict):
        error_metrics_difference_s3_s2 = {}
        for metric_key in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
            if metric_key in s3_model_errors and isinstance(s3_model_errors[metric_key], np.ndarray) and \
               metric_key in s2_model_errors_on_s3 and isinstance(s2_model_errors_on_s3[metric_key], np.ndarray) and \
               s3_model_errors[metric_key].shape == s2_model_errors_on_s3[metric_key].shape:
                difference_grid = s3_model_errors[metric_key] - s2_model_errors_on_s3[metric_key]
                error_metrics_difference_s3_s2[f"Diff_{metric_key}_(S3-S2)"] = difference_grid
            else:
                logger.warning(f"無法計算指標 '{metric_key}' 的差異網格 (S3-S2)，因數據缺失、類型錯誤或形狀不匹配。")
        if error_metrics_difference_s3_s2:
            plot_grid_with_error_long_term(
                dataset_s3_obj_for_eval,
                error_metrics_difference_s3_s2,
                config,
                f"{prefix}_diff_S3_minus_S2"
            )

    # S2 vs BM
    if isinstance(s2_model_errors_on_s3, dict) and isinstance(bm_errors_on_s3, dict):
        error_metrics_difference_s2_bm = {}
        for metric_key in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
            if metric_key in s2_model_errors_on_s3 and isinstance(s2_model_errors_on_s3[metric_key], np.ndarray) and \
               metric_key in bm_errors_on_s3 and isinstance(bm_errors_on_s3[metric_key], np.ndarray) and \
               s2_model_errors_on_s3[metric_key].shape == bm_errors_on_s3[metric_key].shape:
                difference_grid = s2_model_errors_on_s3[metric_key] - bm_errors_on_s3[metric_key]
                error_metrics_difference_s2_bm[f"Diff_{metric_key}_(S2-BM)"] = difference_grid
            else:
                logger.warning(f"無法計算指標 '{metric_key}' 的差異網格 (S2-BM)，因數據缺失、類型錯誤或形狀不匹配。")
        if error_metrics_difference_s2_bm:
            plot_grid_with_error_long_term(
                dataset_s3_obj_for_eval,
                error_metrics_difference_s2_bm,
                config,
                f"{prefix}_diff_S2_minus_BM"
            )

    return results, error_grids_all_models
#%%
if __name__ == '__main__':
    logger.info(f"===== DDPM Stage 2 Training and Evaluation =====")
    logger.info(f"Full CONFIG: {json.dumps(CONFIG, indent=2)}")

    # --- 載入完整數據 ---
    full_df = pd.read_csv(CONFIG["data_path"])
    logger.info(f"已載入資料: {CONFIG['data_path']}. 形狀: {full_df.shape}")

    # === 步驟 1: 載入預訓練的 Basemodel (僅用於生成條件) ===
    # basemodel_for_output_generation 實例將使用其原始的 DDPM3D.sample 邏輯
    # (即接收小時和假日純量，內部轉換為網格)
    BASEMODEL_CHECKPOINT_PATH = CONFIG["basemodel_checkpoint_to_load_for_stage2"]
    if not os.path.exists(BASEMODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"未找到 Basemodel 檢查點: {BASEMODEL_CHECKPOINT_PATH}")

    logger.info(f"===== 載入 Basemodel (for output generation) 從: {BASEMODEL_CHECKPOINT_PATH} =====")
    chkpt_basemodel_eval = torch.load(BASEMODEL_CHECKPOINT_PATH, map_location=CONFIG["device"], weights_only = False)
    if 'ddpm_state_dict' not in chkpt_basemodel_eval:
        raise KeyError(f"Basemodel 檢查點 {BASEMODEL_CHECKPOINT_PATH} 中未找到 'ddpm_state_dict'。")
    if 'selected_sensor_info' in chkpt_basemodel_eval:
        print("'selected_sensor_info' 存在")
    else:
        raise KeyError("'selected_sensor_info' 不存在於檢查點中。")
    
    config_basemodel_original = chkpt_basemodel_eval.get('config', CONFIG)
    CONFIG["cached_basemodel_selected_sensor_info"] = chkpt_basemodel_eval.get('selected_sensor_info')
    CONFIG["cached_basemodel_grid_idx_to_rc_map"] = chkpt_basemodel_eval.get('grid_idx_to_rc_map')
    CONFIG["cached_basemodel_sorted_flow_columns"] = chkpt_basemodel_eval.get('sorted_flow_columns') # 這個已經存在了

    if not CONFIG["cached_basemodel_selected_sensor_info"] or \
    not CONFIG["cached_basemodel_grid_idx_to_rc_map"]:
        raise ValueError("Basemodel 檢查點缺少 'selected_sensor_info' 或 'grid_idx_to_rc_map'。")

    basemodel_unet_for_eval_only = UNet3D(
        config_basemodel_original.get("image_channels", CONFIG["image_channels"]),
        config_basemodel_original.get("base_channels_unet", CONFIG["base_channels_unet"]),
        config_basemodel_original.get("time_emb_dim", CONFIG["time_emb_dim"]),
        config_basemodel_original.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        dropout_rate=config_basemodel_original.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
    ).to(CONFIG["device"])

    basemodel_for_output_generation = DDPM3D(
        basemodel_unet_for_eval_only,
        config_basemodel_original.get("timesteps", CONFIG["timesteps"]),
        (config_basemodel_original.get("D", CONFIG["D"]),
         config_basemodel_original.get("H", CONFIG["H"]),
         config_basemodel_original.get("W", CONFIG["W"])),
        config_basemodel_original.get("image_channels", CONFIG["image_channels"]),
        config_basemodel_original.get("condition_input_channels", CONFIG.get("basemodel_condition_input_channels", 2)),
        config_basemodel_original.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        beta_start=config_basemodel_original.get("beta_start", CONFIG["beta_start"]), # 明確傳遞
        beta_end=config_basemodel_original.get("beta_end", CONFIG["beta_end"]),     # 明確傳遞
        device=CONFIG["device"]  
    )

    basemodel_for_output_generation.load_state_dict(chkpt_basemodel_eval['ddpm_state_dict'])
    basemodel_for_output_generation.eval()
    logger.info(f"Basemodel (for output generation, unified DDPM3D) 載入完成。")

    if 'norm_stats_flow' not in chkpt_basemodel_eval or 'sorted_flow_columns' not in chkpt_basemodel_eval:
        raise ValueError("Basemodel 檢查點必須包含 'norm_stats_flow' 和 'sorted_flow_columns'。")
    basemodel_norm_stats_source = chkpt_basemodel_eval['norm_stats_flow']
    basemodel_sorted_flow_cols_source = chkpt_basemodel_eval['sorted_flow_columns'] 
    CONFIG["cached_basemodel_mean"] = float(basemodel_norm_stats_source['mean'])
    CONFIG["cached_basemodel_std"] = float(basemodel_norm_stats_source['std'])
    if CONFIG["cached_basemodel_std"] < 1e-6: CONFIG["cached_basemodel_std"] = 1.0
    CONFIG["cached_basemodel_sorted_flow_columns"] = basemodel_sorted_flow_cols_source
    CONFIG["cached_basemodel_selected_sensor_info"] = chkpt_basemodel_eval.get('selected_sensor_info')
    CONFIG["cached_basemodel_grid_idx_to_rc_map"] = chkpt_basemodel_eval.get('grid_idx_to_rc_map')

    if not CONFIG.get("cached_basemodel_selected_sensor_info") or \
    not CONFIG.get("cached_basemodel_grid_idx_to_rc_map") or \
    not CONFIG.get("cached_basemodel_sorted_flow_columns"):
        raise ValueError("Basemodel 檢查點缺少必要的網格映射資訊 (selected_sensor_info, grid_idx_to_rc_map, or sorted_flow_columns)。無法繼續。")
    else:
        logger.info("成功從 Basemodel 檢查點加載網格映射資訊到 CONFIG。")
        logger.info("cached_basemodel_mean = {:.4f}".format(CONFIG["cached_basemodel_mean"]))
        logger.info("cached_basemodel_std = {:.4f}".format(CONFIG["cached_basemodel_std"]))
#%%
    # --- 步驟 2: 準備 Stage2 數據 ---
    NEW_COND_FEATURE_COL = CONFIG["stage2_new_condition_feature_column"]
    NEW_COND_OPERATOR = CONFIG["stage2_new_conditional_operator"]
    NEW_COND_VALUE = CONFIG["stage2_new_conditional_value"]
    STAGE2_MODEL_NAME = CONFIG["stage2_model_name"]


    logger.info(f"===== STAGE 2: 數據準備 =====")
    logger.info(f"Stage2 '{NEW_COND_FEATURE_COL} {NEW_COND_OPERATOR} {NEW_COND_VALUE}' 條件劃分的兩個數據分支。")

    df_for_stage2_processing = full_df.copy() # 直接使用完整的 DataFrame

    # --- Basemodel 輸出快取邏輯 ---
    basemodel_outputs_cache_filepath = os.path.join(CONFIG["cache_dir_full_path"], CONFIG["cached_basemodel_outputs_for_s2_filename"])
    all_bm_outputs_s2_np_cond_normalized = None

    if os.path.exists(basemodel_outputs_cache_filepath):
        try:
            logger.info(f"Stage2: 正在從快取檔案載入 Basemodel 輸出: {basemodel_outputs_cache_filepath}")
            all_bm_outputs_s2_np_cond_normalized = np.load(basemodel_outputs_cache_filepath)
            logger.info(f"Stage2: Basemodel 輸出 (正規化) 從快取載入完畢, 形狀: {all_bm_outputs_s2_np_cond_normalized.shape}")
            # 簡單驗證形狀是否合理
            if all_bm_outputs_s2_np_cond_normalized.shape[0] != len(df_for_stage2_processing):
                logger.warning(f"快取的 Basemodel 輸出樣本數 ({all_bm_outputs_s2_np_cond_normalized.shape[0]}) 與預期 ({len(df_for_stage2_processing)}) 不符。將重新生成。")
                all_bm_outputs_s2_np_cond_normalized = None # 標記為需要重新生成
        except Exception as e:
            logger.error(f"Stage2: 從快取檔案 {basemodel_outputs_cache_filepath} 載入 Basemodel 輸出失敗: {e}。將重新生成。")
            all_bm_outputs_s2_np_cond_normalized = None
            
    if all_bm_outputs_s2_np_cond_normalized is None:
        logger.info("Stage2: 生成 basemodel 輸出作為條件...")
        temp_dt_s2_bm_in = pd.to_datetime(df_for_stage2_processing['時間'])
        hours_for_bm_in_s2_scalar = torch.tensor(temp_dt_s2_bm_in.dt.hour.values, dtype=torch.long) # .to(CONFIG["device"])

        if 'holiday' not in df_for_stage2_processing.columns and 'hoilday' in df_for_stage2_processing.columns:
            df_for_stage2_processing.rename(columns={"hoilday": "holiday"}, inplace=True)
        if 'holiday' not in df_for_stage2_processing.columns:
            raise KeyError("在 df_for_stage2_processing 中找不到 'holiday' 欄位。")
        is_holiday_for_bm_in_s2_scalar = torch.tensor(df_for_stage2_processing['holiday'].astype(int).values, dtype=torch.long) # .to(CONFIG["device"])

        bm_outputs_s2_list_cond = []
        pred_bs_s2 = CONFIG.get("eval_batch_size", 64)
        num_s2_samples_for_bm_pred = len(df_for_stage2_processing)

        with torch.no_grad():
            for i in tqdm(range(0, num_s2_samples_for_bm_pred, pred_bs_s2), desc="Basemodel Outputs for S2 Cond"):
                b_hrs_s = hours_for_bm_in_s2_scalar[i:i+pred_bs_s2].to(CONFIG["device"])
                b_hols_s = is_holiday_for_bm_in_s2_scalar[i:i+pred_bs_s2].to(CONFIG["device"])
            
                # 調用 basemodel_for_output_generation.sample
                # basemodel_for_output_generation 的 DDPM3D 實例需要能處理小時/假日純量
                # 假設它的 _prepare_conditional_input_grids (或類似方法) 會將純量轉網格
                bm_pred_norm_b = basemodel_for_output_generation.sample(
                    batch_size=len(b_hrs_s),
                    hour_scalars_batch=b_hrs_s, 
                    is_holiday_scalars_batch=b_hols_s
                )
                bm_pred_denorm_b = bm_pred_norm_b * CONFIG["cached_basemodel_std"] + CONFIG["cached_basemodel_mean"]
                bm_outputs_s2_list_cond.append(bm_pred_denorm_b.cpu().numpy()) # (N, C, D, H, W)
            
        all_bm_outputs_s2_np_cond = np.concatenate(bm_outputs_s2_list_cond, axis=0)
        if all_bm_outputs_s2_np_cond.shape[1] != 1: # 確保條件網格是單通道 (C=1)
            logger.warning(f"Basemodel output for Stage2 conditions has {all_bm_outputs_s2_np_cond.shape[1]} channels, expected 1. Using first channel.")
            all_bm_outputs_s2_np_cond = all_bm_outputs_s2_np_cond[:, 0:1, ...]
        logger.info(f"Stage2: Basemodel 輸出 (條件) 生成完畢, 形狀: {all_bm_outputs_s2_np_cond.shape}")
        
        bm_mean_for_norm = CONFIG.get("cached_basemodel_mean")
        bm_std_for_norm = CONFIG.get("cached_basemodel_std")

        if bm_mean_for_norm is None or bm_std_for_norm is None:
            raise ValueError("CONFIG 中缺少 cached_basemodel_mean 或 cached_basemodel_std，無法正規化 basemodel 的輸出。")
        if bm_std_for_norm < 1e-6: # 避免除以過小的數，與之前邏輯保持一致
            logger.warning(f"Cached basemodel_std ({bm_std_for_norm}) 過小，將其視為 1.0 進行正規化。")
            bm_std_for_norm = 1.0

        all_bm_outputs_s2_np_cond_normalized = (all_bm_outputs_s2_np_cond - bm_mean_for_norm) / bm_std_for_norm
        
        try:
            logger.info(f"Stage2: 正在儲存 Basemodel 輸出到快取檔案: {basemodel_outputs_cache_filepath}")
            np.save(basemodel_outputs_cache_filepath, all_bm_outputs_s2_np_cond_normalized)
            logger.info(f"Stage2: Basemodel 輸出已儲存到快取。")
        except Exception as e:
            logger.error(f"Stage2: 儲存 Basemodel 輸出到快取檔案 {basemodel_outputs_cache_filepath} 失敗: {e}")

    logger.info(f"Stage2: Basemodel 輸出 (條件) 正規化完畢, 形狀: {all_bm_outputs_s2_np_cond_normalized.shape}")
    logger.info(f"正規化後 Basemodel 輸出的均值: {np.mean(all_bm_outputs_s2_np_cond_normalized):.4f}, 標準差: {np.std(all_bm_outputs_s2_np_cond_normalized):.4f}")

    s2_indices_all = np.arange(len(df_for_stage2_processing))
    np.random.shuffle(s2_indices_all) # 使用全局種子
    s2_train_len_final = int(CONFIG["train_split_ratio"] * len(s2_indices_all))
    s2_val_len_final = int(CONFIG["val_split_ratio"] * len(s2_indices_all))

    s2_train_indices_final = s2_indices_all[:s2_train_len_final]
    s2_val_indices_final = s2_indices_all[s2_train_len_final : s2_train_len_final + s2_val_len_final]
    s2_test_indices_final = s2_indices_all[s2_train_len_final + s2_val_len_final:]

    logger.info(f"Stage2 資料分割 (基於篩選後數據): 訓練集={len(s2_train_indices_final)}, 驗證集={len(s2_val_indices_final)}, 測試集={len(s2_test_indices_final)}")

    config_for_s2_dataset_use = CONFIG.copy() # 可能仍被 test_dataset_s3_final 使用




    # --- Stage2 模型最終評估 (實際為載入預訓練模型供後續使用) ---
    logger.info(f"===== STAGE 2: 載入預訓練模型 ({STAGE2_MODEL_NAME}) FOR EVALUATION/STAGE3 =====")
    stage2_model_save_checkpoint_path_full = CONFIG["stage2_checkpoint_full_path"] # 確保此變數已定義

    if not os.path.exists(stage2_model_save_checkpoint_path_full):
        raise FileNotFoundError(f"未找到預訓練的 Stage2 模型檢查點: {stage2_model_save_checkpoint_path_full}. Stage2 模型不進行訓練，必須提供檢查點。")

    logger.info(f"從 {stage2_model_save_checkpoint_path_full} 載入預訓練的 Stage2 模型...")
    chkpt_s2_final_for_eval = torch.load(stage2_model_save_checkpoint_path_full, map_location=CONFIG["device"], weights_only=False)

    # 使用儲存在檢查點中的配置（如果存在）或當前配置來初始化模型結構
    config_from_s2_chkpt_eval = chkpt_s2_final_for_eval.get('config_snapshot_at_save', CONFIG)

    eval_s2_unet_final = UNet3D(
        config_from_s2_chkpt_eval.get("image_channels", CONFIG["image_channels"]),
        config_from_s2_chkpt_eval.get("base_channels_unet", CONFIG["base_channels_unet"]),
        config_from_s2_chkpt_eval.get("time_emb_dim", CONFIG["time_emb_dim"]),
        config_from_s2_chkpt_eval.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        dropout_rate=config_from_s2_chkpt_eval.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
    ).to(CONFIG["device"])

    final_s2_model_to_eval = DDPM3D(
        unet_model=eval_s2_unet_final,
        timesteps=config_from_s2_chkpt_eval.get("timesteps", CONFIG["timesteps"]),
        image_size=(config_from_s2_chkpt_eval.get("D", CONFIG["D"]), config_from_s2_chkpt_eval.get("H", CONFIG["H"]), config_from_s2_chkpt_eval.get("W", CONFIG["W"])),
        image_channels=config_from_s2_chkpt_eval.get("image_channels", CONFIG["image_channels"]),
        condition_input_channels=config_from_s2_chkpt_eval.get("stage2_ddpm_condition_input_channels", CONFIG.get("stage2_ddpm_condition_input_channels", 2)),
        condition_encode_dim=config_from_s2_chkpt_eval.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        beta_start=config_from_s2_chkpt_eval.get("beta_start", CONFIG["beta_start"]),
        beta_end=config_from_s2_chkpt_eval.get("beta_end", CONFIG["beta_end"]),
        device=CONFIG["device"]
    )
    final_s2_model_to_eval.load_state_dict(chkpt_s2_final_for_eval['ddpm_state_dict'])
    logger.info(f"最佳 Stage2 模型 (Epoch {chkpt_s2_final_for_eval.get('epoch','未知')}) 載入完成。")
    s2_avg_flow_map_for_final_eval = chkpt_s2_final_for_eval.get('stage2_avg_flow_map_dict')
    new_cond_feature_norm_stats_for_final_eval = chkpt_s2_final_for_eval.get('new_cond_feature_norm_stats')
    stage2_target_norm_stats_for_final_eval = chkpt_s2_final_for_eval.get('norm_stats_stage2_target') # 從檢查點獲取   

    if s2_avg_flow_map_for_final_eval is None or new_cond_feature_norm_stats_for_final_eval is None or \
       stage2_target_norm_stats_for_final_eval is None:
         raise ValueError("無法從 Stage2 檢查點獲取必要的統計量 (avg_flow_map, new_cond_norm_stats, 或 stage2_target_norm_stats)。請確保檢查點包含這些資訊。")

    # 準備測試集 Loader (此部分命名混淆，實際上可能用於 Stage3 評估)
    # 注意: test_loader_s2_final 的創建依賴 s2_test_indices_final, df_for_stage2_processing, all_bm_outputs_s2_np_cond_normalized
    # 以及從 S2 檢查點載入的統計數據。
    # Stage2Dataset 類別的 __getitem__ 實際上返回的是 Stage3 的目標和條件。
    test_loader_s2_final = None
    if len(s2_test_indices_final) > 0:
            # test_dataset_s3_final 的命名也暗示了它用於 Stage3
            # 但它使用了 Stage2Dataset 類別，並傳入了 Stage2 相關的數據和統計量
            # 這裡的 df_for_stage2_processing 和 all_bm_outputs_s2_np_cond_normalized 是 Stage2 的數據
            # 這部分邏輯如果用於 Stage3 評估，其數據流是混亂的，但我們僅移除 Stage 2 訓練。
            test_dataset_s3_final = Stage2Dataset( 
                df_for_stage2_processing=df_for_stage2_processing.iloc[s2_test_indices_final], # 使用 Stage2 的 DataFrame 子集
                basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_test_indices_final], # 使用 Basemodel 為 Stage2 生成的輸出
                config=config_for_s2_dataset_use, # CONFIG 的副本
                mode='test',
                original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
                # 以下統計數據來自載入的 Stage2 檢查點
                s2_new_cond_feature_norm_stats_from_s2_train=new_cond_feature_norm_stats_for_final_eval, 
                # s3_new_cond_feature_norm_stats_from_train # Stage2Dataset不需要這個
                stage3_target_norm_stats_from_train=stage2_target_norm_stats_for_final_eval # Stage2Dataset 用這個作為 S2 目標統計
            )
            s2_eval_batch_size_final = CONFIG.get("eval_batch_size")
            test_loader_s2_final = DataLoader(test_dataset_s3_final, batch_size=s2_eval_batch_size_final, shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
            logger.info(f"為後續階段準備的測試數據集 (基於Stage2數據和模型) 創建，含 {len(test_dataset_s3_final)} 樣本。")
    else:
            logger.info("Stage2 測試集 (s2_test_indices_final) 為空，無法創建 test_loader_s2_final。")

    # --- 步驟 S3-4: 最終評估 Stage3 模型 ---
    # 載入最佳 Stage3 模型
    # 呼叫 evaluate_stage3_models(...)
    # 儲存結果

    logger.info(f"===== STAGE 3: 最終評估 =====")
    # 假設 stage3_model 是訓練好的或已載入的 Stage3 模型
    # 假設 final_s2_model_to_eval 是已載入的 Stage2 模型
    # 假設 basemodel_for_output_generation 是已載入的 Basemodel
    # 假設 test_loader_s2_final 是 Stage3 的測試數據加載器 (命名可能混淆，但它包含S3目標和所有必要條件)
    
    # 實例化 Inception 模型 (如果尚未實例化)
    inception_model_for_fid_final_eval = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
    inception_model_for_fid_final_eval.fc = nn.Identity() # 移除最後一層
    inception_model_for_fid_final_eval = inception_model_for_fid_final_eval.to(CONFIG["device"])
    inception_model_for_fid_final_eval.eval()


    # 決定要評估哪個 Stage3 模型 (是剛訓練的，還是從檢查點載入的)
    s3_model_to_finally_evaluate = None
    if 'stage3_model' in locals() and stage3_model is not None: # 如果剛訓練了 Stage3 模型
        logger.info("將評估剛訓練的 Stage3 模型。")
        s3_model_to_finally_evaluate = stage3_model
    elif os.path.exists(CONFIG["stage3_checkpoint_full_path"]):
        logger.info(f"從檢查點 {CONFIG['stage3_checkpoint_full_path']} 載入 Stage3 模型進行最終評估。")
        chkpt_s3_final_eval = torch.load(CONFIG["stage3_checkpoint_full_path"], map_location=CONFIG["device"], weights_only=False)
        config_from_s3_chkpt_eval = chkpt_s3_final_eval.get('config_snapshot_at_save', CONFIG)
        
        eval_s3_unet_final = UNet3D(
            config_from_s3_chkpt_eval.get("image_channels", CONFIG["image_channels"]),
            config_from_s3_chkpt_eval.get("base_channels_unet", CONFIG["base_channels_unet"]),
            config_from_s3_chkpt_eval.get("time_emb_dim", CONFIG["time_emb_dim"]),
            config_from_s3_chkpt_eval.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
            dropout_rate=config_from_s3_chkpt_eval.get("unet_dropout_rate", CONFIG.get("unet_dropout_rate", 0.05))
        ).to(CONFIG["device"])

        s3_model_to_finally_evaluate = DDPM3D(
            unet_model=eval_s3_unet_final,
            timesteps=config_from_s3_chkpt_eval.get("timesteps", CONFIG["timesteps"]),
            image_size=(config_from_s3_chkpt_eval.get("D", CONFIG["D"]), config_from_s3_chkpt_eval.get("H", CONFIG["H"]), config_from_s3_chkpt_eval.get("W", CONFIG["W"])),
            image_channels=config_from_s3_chkpt_eval.get("image_channels", CONFIG["image_channels"]),
            condition_input_channels=config_from_s3_chkpt_eval.get("stage3_ddpm_condition_input_channels", CONFIG.get("stage3_ddpm_condition_input_channels", 2)),
            condition_encode_dim=config_from_s3_chkpt_eval.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
            beta_start=config_from_s3_chkpt_eval.get("beta_start", CONFIG["beta_start"]),
            beta_end=config_from_s3_chkpt_eval.get("beta_end", CONFIG["beta_end"]),
            device=CONFIG["device"]
        )
        s3_model_to_finally_evaluate.load_state_dict(chkpt_s3_final_eval['ddpm_state_dict'])
        logger.info(f"從檢查點載入的 Stage3 模型 (Epoch {chkpt_s3_final_eval.get('epoch','未知')}) 已準備好進行評估。")
    else:
        logger.error("沒有可用的 Stage3 模型進行最終評估")
        s3_model_to_finally_evaluate = None


    if s3_model_to_finally_evaluate and final_s2_model_to_eval and basemodel_for_output_generation and test_loader_s2_final:
        final_s3_metrics, final_s3_error_grids = evaluate_stage3_models(
            stage3_model_trained=s3_model_to_finally_evaluate,
            stage2_model_eval_instance=final_s2_model_to_eval, # Pass loaded Stage 2 model
            basemodel_eval_instance=basemodel_for_output_generation,
            dataloader_s3=test_loader_s2_final, # This is the test loader for S3 targets
            inception_model_fid=inception_model_for_fid_final_eval,
            config=CONFIG,
            max_samples_for_fid=CONFIG.get("fid_num_samples"),
            prefix="final_s3_evaluation"
        )
        logger.info(f"Stage3 最終評估指標: {json.dumps(final_s3_metrics, indent=2)}")
        
        # 儲存指標
        metrics_save_path = os.path.join(CONFIG["stage3_model_save_dir"], "final_stage3_evaluation_metrics.json")
        with open(metrics_save_path, 'w') as f:
            json.dump(final_s3_metrics, f, indent=4)
        logger.info(f"Stage3 最終評估指標已儲存到: {metrics_save_path}")

        # 可以選擇性地儲存 error_grids (可能是 .npz 檔案)
        error_grids_save_path = os.path.join(CONFIG["stage3_model_save_dir"], "final_stage3_evaluation_error_grids.npz")
        # 轉換 error_grids_all_models 中的 numpy 數組以便儲存
        error_grids_to_save = {}
        for model_key, metric_dict in final_s3_error_grids.items():
            for metric_name, grid_array in metric_dict.items():
                error_grids_to_save[f"{model_key}_{metric_name}"] = grid_array
        try:
            np.savez_compressed(error_grids_save_path, **error_grids_to_save)
            logger.info(f"Stage3 最終評估誤差網格已儲存到: {error_grids_save_path}")
        except Exception as e:
            logger.error(f"儲存 Stage3 誤差網格失敗: {e}")

    else:
        logger.warning("由於缺少必要的模型或數據加載器，跳過 Stage3 最終評估。")

    logger.info("===== DDPM Stage3 流程結束 =====")

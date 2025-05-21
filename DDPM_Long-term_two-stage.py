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

    # --- 模型架構參數 (basemodel 和 stage2_model 共用) ---
    "image_channels": 1,      # 主要資料(流量圖)的通道數
    "base_channels_unet": 64,   # UNet3D 的基礎通道數
    "unet_dropout_rate": 0.1,
    "time_emb_dim": 256,        # 時間嵌入維度
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 / UNet中與x_t合併的維度

    # === Basemodel 相關 (用於載入並決定其原始條件處理方式) ===
    # Basemodel 的 condition_processor 輸入通道數 (通常是2，因為它內部將小時、假日純量轉為2個網格)
    "basemodel_condition_input_channels": 2, # 假設原始basemodel用2通道條件(小時網格+假日網格)
    "basemodel_checkpoint_to_load_for_stage2": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_long-term\best_ddpm_model_during_training.pth",

    # === Stage2 特定配置 ===
    "stage2_new_condition_feature_column": "露點溫度", # 新條件的欄位名
    "stage2_new_conditional_operator": "<=",         # 新條件的運算符
    "stage2_new_conditional_value": 23.5,             # 新條件的閾值
    "stage2_model_name": "Stage2_dew_point_le_23_5",    # 第二階段模型的名稱
    "stage2_ddpm_condition_input_channels": 2,       # Stage2 DDPM 的 condition_processor 輸入通道數 (固定為2: bm_out + uv_grid)
    "stage2_checkpoint_path": "best_stage2_model_dew_point_le_23_5.pth", # Stage2 模型的檢查點檔名 (相對路徑，相對於stage2_model_save_dir)

    # --- DDPM 擴散參數 ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 訓練參數 (Stage2 將優先使用 epochs_stage2, lr_stage2 等，若無則回退到通用版本) ---
    "epochs": 128, 
    "batch_size": 128,
    "lr": 1e-3, 
    
    "epochs_stage2": 128, # 可為 Stage2 設定不同的 epoch 數
    "lr_stage2": 1e-3,   # 可為 Stage2 設定不同的學習率

    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "weight_decay": 1e-5,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 3,
    "lr_scheduler_min_lr": 1e-6, 
    "early_stopping_patience": 6,
    "val_calculation_freq": 4,  

    "resume_from_stage2_checkpoint": True,  # Stage2 訓練是否從自己的檢查點續訓

    # --- 評估參數 ---
    "eval_batch_size": 32,
    "fid_batch_size": 64,
    "fid_num_samples": 128, # 通用FID樣本數
    "fid_num_samples_stage2": 128, # Stage2 FID 計算樣本數 (可與通用相同或不同)


    # --- 路徑與儲存 ---
    "save_dir": "results_ddpm_stage2", # 主結果儲存目錄的基礎名稱
    "plot_grid_mapping_path_stage2": "grid_mapping_visualization_stage2.png",
    
    "train_split_ratio": 0.7,
    "val_split_ratio": 0.15,
}

CONFIG["condition_input_channels"] = CONFIG.get("stage2_ddpm_condition_input_channels", 2)

# 更新/生成 Stage2 相關路徑
CONFIG["stage2_model_save_dir"] = os.path.join(CONFIG["save_dir"], CONFIG["stage2_model_name"])
os.makedirs(CONFIG["stage2_model_save_dir"], exist_ok=True) # 確保主 save_dir 也創建
os.makedirs(CONFIG["stage2_model_save_dir"], exist_ok=True)
CONFIG["stage2_checkpoint_full_path"] = os.path.join(CONFIG["stage2_model_save_dir"], CONFIG["stage2_checkpoint_path"])

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
        if condition_grid_1_batch.shape[1:] != expected_single_grid_shape or \
           condition_grid_2_batch.shape[1:] != expected_single_grid_shape:
            self.logger.error(f"Stage 2 condition input grid dimensions are incorrect. Grid1: {condition_grid_1_batch.shape}, Grid2: {condition_grid_2_batch.shape}. Expected N,1,D,H,W")
            # Consider raising an error if shapes are critical and cannot be recovered.
        
        # 這裡假設 condition_processor 的輸入通道數是 2
        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_stage2_condition_grids: Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels.")
        return torch.cat((condition_grid_1_batch, condition_grid_2_batch), dim=1)

    def p_losses(self, x_start_target_flow: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None,
                 # 條件參數 - 擇一提供
                 hour_scalars_batch: Optional[torch.Tensor] = None,
                 is_holiday_scalars_batch: Optional[torch.Tensor] = None,
                 basemodel_output_grid_batch: Optional[torch.Tensor] = None,
                 new_condition_feature_grid_batch: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:

        if noise is None: noise = torch.randn_like(x_start_target_flow)
        x_t_noisy_target = self.q_sample(x_start=x_start_target_flow, t=t, noise=noise)

        stacked_cond_grids: Optional[torch.Tensor] = None
        # 判斷是哪種條件模式
        if hour_scalars_batch is not None and is_holiday_scalars_batch is not None:
            # Basemodel (原始) 條件模式
            if basemodel_output_grid_batch is not None or new_condition_feature_grid_batch is not None:
                raise ValueError("p_losses: Cannot provide both scalar (hour/holiday) and grid conditions simultaneously.")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(
                hour_scalars_batch, is_holiday_scalars_batch
            )
        elif basemodel_output_grid_batch is not None and new_condition_feature_grid_batch is not None:
            # Stage2 條件模式
            stacked_cond_grids = self._prepare_stage2_condition_grids(
                basemodel_output_grid_batch,
                new_condition_feature_grid_batch
            )
        else:
            raise ValueError("p_losses: Insufficient or ambiguous condition arguments provided.")

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
               hour_scalars_batch: Optional[torch.Tensor] = None,
               is_holiday_scalars_batch: Optional[torch.Tensor] = None,
               basemodel_output_grid_batch: Optional[torch.Tensor] = None,
               new_condition_feature_grid_batch: Optional[torch.Tensor] = None
               ) -> torch.Tensor:

        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)

        stacked_cond_grids: Optional[torch.Tensor] = None
        if hour_scalars_batch is not None and is_holiday_scalars_batch is not None:
            # Basemodel (原始) 條件模式
            if basemodel_output_grid_batch is not None or new_condition_feature_grid_batch is not None:
                raise ValueError("sample: Cannot provide both scalar (hour/holiday) and grid conditions simultaneously.")
            if hour_scalars_batch.shape[0] != batch_size or is_holiday_scalars_batch.shape[0] != batch_size:
                raise ValueError(f"Original condition batch sizes ({hour_scalars_batch.shape[0]},{is_holiday_scalars_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(
                hour_scalars_batch, is_holiday_scalars_batch
            ).to(self.device)
        elif basemodel_output_grid_batch is not None and new_condition_feature_grid_batch is not None:
            if basemodel_output_grid_batch.shape[0] != batch_size or new_condition_feature_grid_batch.shape[0] != batch_size:
                raise ValueError(f"Stage2 condition batch sizes ({basemodel_output_grid_batch.shape[0]},{new_condition_feature_grid_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_stage2_condition_grids(
                basemodel_output_grid_batch,
                new_condition_feature_grid_batch
            )
        else:
            raise ValueError("sample: Insufficient or ambiguous condition arguments provided.")

        # 驗證 condition_processor 的輸入通道數
        expected_cond_proc_input_channels = self.condition_processor[0].in_channels
        if stacked_cond_grids.shape[1] != expected_cond_proc_input_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for sampling. "
                              f"ConditionProcessor expected {expected_cond_proc_input_channels} channels, "
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
    if 'selected_sensor_info' in chkpt_basemodel:
        print("selected_sensor_info 存在")
    else:
        raise KeyError("'selected_sensor_info' 不存在於檢查點中。")
    
    basemodel_original_config = chkpt_basemodel.get('config', config_for_stage2_model)

    # Stage2 模型的 UNet 架構應與 basemodel 一致
    stage2_unet = UNet3D(
        input_image_channels=basemodel_original_config.get("image_channels", config_for_stage2_model["image_channels"]),
        base_channels=basemodel_original_config.get("base_channels_unet", config_for_stage2_model["base_channels_unet"]),
        time_emb_dim=basemodel_original_config.get("time_emb_dim", config_for_stage2_model["time_emb_dim"]),
        condition_encode_dim=basemodel_original_config.get("condition_encode_dim", config_for_stage2_model["condition_encode_dim"]),
        dropout_rate=basemodel_original_config.get("unet_dropout_rate", config_for_stage2_model.get("unet_dropout_rate", 0.05))
    ).to(device)

    stage2_model_condition_input_channels = config_for_stage2_model.get("condition_input_channels")
    if stage2_model_condition_input_channels is None: # 如果主CONFIG沒有被stage2特定值覆蓋
        stage2_model_condition_input_channels = CONFIG.get("stage2_ddpm_condition_input_channels", 2) # 從 Stage2特定配置取
        logger.info(f"Stage2 模型 condition_input_channels 未在傳入的 config_for_stage2_model 中明確指定，"
                    f"將使用 CONFIG['stage2_ddpm_condition_input_channels'] (值: {stage2_model_condition_input_channels})。")

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
        stage2_model_instance.load_state_dict(chkpt_basemodel['ddpm_state_dict'])
        logger.info("Stage2 模型權重從 Basemodel 完整遷移完成。")
    except RuntimeError as e:
        logger.error(f"直接載入 Basemodel state_dict 到 Stage2 模型失敗: {e}")
        logger.warning("嘗試僅載入 UNet (model) 部分的權重...")
        stage2_model_instance.model.load_state_dict(chkpt_basemodel['ddpm_state_dict']['model']) # 假設鍵是 'model'
        logger.info("僅 UNet 權重從 Basemodel 遷移完成。Condition Processor 將使用隨機初始化權重。")

    return stage2_model_instance

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
                 basemodel_outputs_for_samples_np: np.ndarray,
                 config: Dict[str, Any],
                 original_sorted_flow_columns: List[str],
                 mode: str = 'train',
                 stage2_avg_flow_map_dict_from_train: Optional[Dict[Tuple, np.ndarray]] = None,
                 new_cond_feature_norm_stats_from_train: Optional[Dict[str, float]] = None,
                 stage2_target_norm_stats_from_train: Optional[Dict[str, float]] = None
                 ):
        super().__init__()
        self.df_s2 = df_for_stage2_processing.reset_index(drop=True)
        self.basemodel_outputs_np = basemodel_outputs_for_samples_np
        self.config = config 
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        self.H = config["H"]
        self.W = config["W"]
        self.D = config.get("D", 1)
        self.image_channels_target = config.get("image_channels", 1)

        self.new_cond_col_name = config["stage2_new_condition_feature_column"]
        self.new_cond_op = config["stage2_new_conditional_operator"]
        self.new_cond_val = config["stage2_new_conditional_value"]
        self.sorted_flow_columns = original_sorted_flow_columns

        if self.new_cond_col_name not in self.df_s2.columns:
             self.logger.error(f"Stage2Dataset: 新條件特徵的原始欄位 '{self.new_cond_col_name}' "
                               f"不在 DataFrame 的欄位 '{list(self.df_s2.columns)}' 中。")
             raise ValueError(f"Stage2Dataset: 新條件特徵的原始欄位 '{self.new_cond_col_name}' 不在 DataFrame 中。")

        if '時' not in self.df_s2.columns:
            raise KeyError("DataFrame 中找不到 '時' 欄位。")
        dt_series_for_base_cond = pd.to_datetime(self.df_s2['時'])
        self.hours_for_target_np = dt_series_for_base_cond.dt.hour.values
        self.hour_category_for_target_grouping_np = (self.hours_for_target_np > 8).astype(int)
        self.logger.info(f"Stage2Dataset (mode={self.mode}): 已生成用於目標分組的 Base Model 小時類別。 "
                        f"類別0 (hr <= 8) 數量: {np.sum(self.hour_category_for_target_grouping_np == 0)}, "
                        f"類別1 (hr > 8) 數量: {np.sum(self.hour_category_for_target_grouping_np == 1)}")

        if 'holiday' not in self.df_s2.columns and 'hoilday' in self.df_s2.columns:
            self.df_s2.rename(columns={"hoilday": "holiday"}, inplace=True)
        if 'holiday' not in self.df_s2.columns:
            self.logger.error(f"Stage2Dataset (mode={self.mode}): DataFrame 中缺少 'holiday' 欄位。")
            raise KeyError("DataFrame 中找不到 'holiday' 或 'hoilday' 欄位。")
        if self.df_s2['holiday'].dtype == bool:
            self.is_holiday_for_target_np = self.df_s2['holiday'].astype(int).values
        elif pd.api.types.is_numeric_dtype(self.df_s2['holiday']):
             self.is_holiday_for_target_np = self.df_s2['holiday'].fillna(0).astype(bool).astype(int).values
        else:
            holiday_map = {'是': 1, 'true': 1, '1': 1, 'yes': 1, 'y': 1,
                           '否': 0, 'false': 0, '0': 0, 'no': 0, 'n': 0}
            self.is_holiday_for_target_np = self.df_s2['holiday'].astype(str).str.lower().map(holiday_map).fillna(0).astype(int).values
        self.logger.info(f"Stage2Dataset (mode={self.mode}): Base Model 假日欄位處理完畢。")

        self.new_cond_original_values_np = pd.to_numeric(self.df_s2[self.new_cond_col_name], errors='coerce').values
        num_nan_new_cond = np.isnan(self.new_cond_original_values_np).sum()
        if num_nan_new_cond > 0:
            self.logger.warning(f"Stage2Dataset (mode={self.mode}): 新條件欄位 '{self.new_cond_col_name}' "
                               f"包含 {num_nan_new_cond} 個 NaN 值。"
                               "建議預處理。比較時 NaN 通常結果為 False，可能導致歸入分支1。")

        numeric_new_cond_vals_for_category = pd.Series(self.new_cond_original_values_np)
        try:
            threshold_val = float(self.new_cond_val)
            cat_0_desc, cat_1_desc = "", ""
            if self.new_cond_op == "<=":
                condition_met_mask = (numeric_new_cond_vals_for_category <= threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' <= {threshold_val}"
                cat_1_desc = f"'{self.new_cond_col_name}' > {threshold_val}"
            elif self.new_cond_op == ">":
                condition_met_mask = (numeric_new_cond_vals_for_category > threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' > {threshold_val}"
                cat_1_desc = f"'{self.new_cond_col_name}' <= {threshold_val}"
            elif self.new_cond_op == "<":
                condition_met_mask = (numeric_new_cond_vals_for_category < threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' < {threshold_val}"
                cat_1_desc = f"'{self.new_cond_col_name}' >= {threshold_val}"
            elif self.new_cond_op == ">=":
                condition_met_mask = (numeric_new_cond_vals_for_category >= threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' >= {threshold_val}"
                cat_1_desc = f"'{self.new_cond_col_name}' < {threshold_val}"
            elif self.new_cond_op == "==":
                condition_met_mask = (numeric_new_cond_vals_for_category == threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' == {threshold_val}"
                cat_1_desc = f"'{self.new_cond_col_name}' != {threshold_val}"
            elif self.new_cond_op == "!=":
                condition_met_mask = (numeric_new_cond_vals_for_category != threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' != {threshold_val}"
                cat_1_desc = f"'{self.new_cond_col_name}' == {threshold_val}"
            else:
                self.logger.warning(f"Stage2Dataset (mode={self.mode}): 未明確處理新條件運算符 '{self.new_cond_op}' for column '{self.new_cond_col_name}'，"
                                   f"預設分類為 ({self.new_cond_col_name} <= {threshold_val}) 為類別0，否則為類別1。")
                condition_met_mask = (numeric_new_cond_vals_for_category <= threshold_val)
                self.new_cond_category_for_target_np = (~condition_met_mask).astype(int)
                cat_0_desc = f"'{self.new_cond_col_name}' <= {threshold_val} (預設)"
                cat_1_desc = f"'{self.new_cond_col_name}' > {threshold_val} (預設)"
            self.logger.info(f"Stage2Dataset (mode={self.mode}): 新條件 ('{self.new_cond_col_name}') 分類邏輯 -> 類別0 (主要條件滿足): {cat_0_desc}; 類別1 (不滿足): {cat_1_desc}")
        except ValueError:
            self.logger.error(f"Stage2Dataset (mode={self.mode}): 新條件的閾值 '{self.new_cond_val}' 無法轉換為浮點數。請檢查CONFIG。")
            self.new_cond_category_for_target_np = np.zeros(len(self.df_s2), dtype=int)
            self.logger.warning(f"Stage2Dataset (mode={self.mode}): 由於閾值轉換錯誤，所有樣本的新條件 ('{self.new_cond_col_name}') 類別將被設為0。")
        unique_cats, counts_cats = np.unique(self.new_cond_category_for_target_np, return_counts=True)
        self.logger.info(f"Stage2Dataset (mode={self.mode}): 生成的新條件分類 (new_cond_category_for_target_np based on '{self.new_cond_col_name}') 分佈: {dict(zip(unique_cats, counts_cats))}")

        # --- 根據模式(mode)處理正規化統計量和平均流量圖 ---
        if self.mode == 'train':
            # 計算新條件特徵的正規化統計量
            new_cond_values_for_norm_calc = self.new_cond_original_values_np[~np.isnan(self.new_cond_original_values_np)]
            if len(new_cond_values_for_norm_calc) > 0:
                self.new_cond_feature_mean = np.mean(new_cond_values_for_norm_calc)
                self.new_cond_feature_std = np.std(new_cond_values_for_norm_calc)
            else:
                self.logger.warning(f"Stage2Dataset (train): 新條件特徵欄位 '{self.new_cond_col_name}' 中沒有有效的數值 (移除了NaN後) 用於計算正規化統計量。將使用 mean=0, std=1。")
                self.new_cond_feature_mean = 0.0
                self.new_cond_feature_std = 1.0
            if self.new_cond_feature_std < 1e-6:
                self.logger.warning(f"Stage2Dataset (train): 計算得到的新條件特徵 ({self.new_cond_col_name}) 標準差 ({self.new_cond_feature_std:.4f}) 過小，將其設為 1.0。")
                self.new_cond_feature_std = 1.0
            self.norm_stats_new_cond_feature = {'mean': self.new_cond_feature_mean, 'std': self.new_cond_feature_std}
            self.logger.info(f"Stage2Dataset (train): 計算得到新的條件特徵 ({self.new_cond_col_name}) 的正規化統計量: mean={self.new_cond_feature_mean:.4f}, std={self.new_cond_feature_std:.4f}")

            self.logger.info(f"Stage2Dataset (train): 呼叫 _calculate_stage2_target_flows 來設定 self.average_flow_map_dict_s2.")
            calculated_map = self._calculate_stage2_target_flows() # 內部會使用 self.new_cond_category_for_target_np
            self.logger.info(f"Stage2Dataset (train): _calculate_stage2_target_flows() 返回。類型: {type(calculated_map)}")
            if calculated_map is None: self.logger.error("CRITICAL ERROR in Stage2Dataset (train): _calculate_stage2_target_flows() 返回 None!"); self.average_flow_map_dict_s2 = {}
            elif not isinstance(calculated_map, dict): self.logger.error(f"CRITICAL ERROR in Stage2Dataset (train): _calculate_stage2_target_flows() 返回類型 {type(calculated_map)}, 應為 dict! 設定為空字典。"); self.average_flow_map_dict_s2 = {}
            elif not calculated_map: self.logger.warning("Stage2Dataset (train): _calculate_stage2_target_flows() 返回一個空字典。"); self.average_flow_map_dict_s2 = calculated_map
            else: self.logger.info(f"Stage2Dataset (train): _calculate_stage2_target_flows() 返回一個包含 {len(calculated_map)} 個條目的字典。"); self.average_flow_map_dict_s2 = calculated_map
            if hasattr(self, 'average_flow_map_dict_s2') and self.average_flow_map_dict_s2 and isinstance(self.average_flow_map_dict_s2, dict):
                 self.logger.info(f"INFO (train): self.average_flow_map_dict_s2 已正確設定，包含 {len(self.average_flow_map_dict_s2)} 個條目。")
            else: self.logger.error(f"CRITICAL ERROR (train): self.average_flow_map_dict_s2 在賦值後狀態不正確。")


            if self.average_flow_map_dict_s2:
                all_target_flow_maps = np.array(list(self.average_flow_map_dict_s2.values()))
                if all_target_flow_maps.size > 0:
                    s2_target_mean = np.mean(all_target_flow_maps)
                    s2_target_std = np.std(all_target_flow_maps)
                    if s2_target_std < 1e-6:
                        self.logger.warning(f"Stage2Dataset (train): 計算得到的 Stage2 目標專用標準差 ({s2_target_std:.4f}) 過小，將其設為 1.0。")
                        s2_target_std = 1.0
                    self.norm_stats_stage2_target = {'mean': s2_target_mean, 'std': s2_target_std}
                    self.logger.info(f"Stage2Dataset (train): 計算得到 Stage2 目標流量的專用正規化統計量: mean={s2_target_mean:.4f}, std={s2_target_std:.4f}")
                else:
                    self.logger.warning("Stage2Dataset (train): average_flow_map_dict_s2 中的值為空數組，無法計算 Stage2 目標專用統計量。使用預設值。")
                    self.norm_stats_stage2_target = {'mean': 0.0, 'std': 1.0}
            else:
                self.logger.warning("Stage2Dataset (train): average_flow_map_dict_s2 為空，無法計算 Stage2 目標專用統計量。使用預設值。")
                self.norm_stats_stage2_target = {'mean': 0.0, 'std': 1.0}


        elif self.mode == 'val' or self.mode == 'test':
            if new_cond_feature_norm_stats_from_train is None:
                self.logger.error(f"Stage2 {self.mode} mode: new_cond_feature_norm_stats_from_train is None.")
                raise ValueError(f"Stage2 {self.mode} mode 需要從訓練集傳入 new_cond_feature_norm_stats。")
            self.norm_stats_new_cond_feature = new_cond_feature_norm_stats_from_train
            self.new_cond_feature_mean = self.norm_stats_new_cond_feature.get('mean', 0.0)
            self.new_cond_feature_std = self.norm_stats_new_cond_feature.get('std', 1.0)
            if self.new_cond_feature_std < 1e-6:
                self.logger.warning(f"Stage2Dataset ({self.mode}): 從訓練集傳入的新條件 ({self.new_cond_col_name}) 標準差 ({self.new_cond_feature_std:.4f}) 過小或無效，將其設為 1.0。")
                self.new_cond_feature_std = 1.0
            self.logger.info(f"Stage2Dataset ({self.mode}): 已載入新的條件特徵 ({self.new_cond_col_name}) 的正規化統計量: mean={self.new_cond_feature_mean:.4f}, std={self.new_cond_feature_std:.4f}")

            if stage2_target_norm_stats_from_train is None:
                self.logger.error(f"Stage2 {self.mode} mode: use_dedicated_stage2_target_norm 為 True 但 stage2_target_norm_stats_from_train is None.")
                raise ValueError(f"Stage2 {self.mode} mode 需要從訓練集傳入 stage2_target_norm_stats。")
            self.norm_stats_stage2_target = stage2_target_norm_stats_from_train
            s2_target_mean = self.norm_stats_stage2_target.get('mean', 0.0)
            s2_target_std = self.norm_stats_stage2_target.get('std', 1.0)
            if s2_target_std < 1e-6:
                self.logger.warning(f"Stage2Dataset ({self.mode}): 從訓練集傳入的 Stage2 目標專用標準差 ({s2_target_std:.4f}) 過小或無效，將其設為 1.0。")
            self.logger.info(f"Stage2Dataset ({self.mode}): 已載入 Stage2 目標流量的專用正規化統計量: mean={s2_target_mean:.4f}, std={s2_target_std:.4f}") # s2_target_std 在 .get() 時已經處理了


            if stage2_avg_flow_map_dict_from_train is None:
                self.logger.error(f"Stage2 {self.mode} mode: stage2_avg_flow_map_dict_from_train is None.")
                raise ValueError(f"Stage2 {self.mode} mode 需要從訓練集傳入 stage2_avg_flow_map_dict。")
            if not isinstance(stage2_avg_flow_map_dict_from_train, dict):
                self.logger.error(f"Stage2 {self.mode} mode: stage2_avg_flow_map_dict_from_train 類型為 {type(stage2_avg_flow_map_dict_from_train)}, 應為 dict.")
                raise TypeError(f"Stage2 {self.mode} mode: stage2_avg_flow_map_dict_from_train 必須是字典。")
            self.average_flow_map_dict_s2 = stage2_avg_flow_map_dict_from_train
            self.logger.info(f"INFO ({self.mode} mode): self.average_flow_map_dict_s2 已從傳入參數賦值。Is None: {self.average_flow_map_dict_s2 is None}. "
                             f"Length if dict: {len(self.average_flow_map_dict_s2) if isinstance(self.average_flow_map_dict_s2, dict) else 'N/A'}")
        else:
            self.logger.error(f"未知的 Stage2Dataset mode: {self.mode}")
            raise ValueError(f"未知的 Stage2Dataset mode: {self.mode}")

        final_check_attr_name = 'average_flow_map_dict_s2'
        if not hasattr(self, final_check_attr_name):
             self.logger.error(f"CRITICAL ERROR FINAL CHECK (mode={self.mode}): 屬性 '{final_check_attr_name}' 在 __init__ 結束時不存在!")
        else: 
            attr_value = getattr(self, final_check_attr_name)
            if attr_value is None: self.logger.error(f"CRITICAL FINAL CHECK (mode={self.mode}): 屬性 '{final_check_attr_name}' 在 __init__ 結束時為 None!")
            elif not isinstance(attr_value, dict): self.logger.error(f"CRITICAL FINAL CHECK (mode={self.mode}): 屬性 '{final_check_attr_name}' 在 __init__ 結束時類型為 {type(attr_value)}, 不是 dict!")
            elif not attr_value and isinstance(attr_value, dict): self.logger.warning(f"FINAL CHECK (mode={self.mode}): 屬性 '{final_check_attr_name}' 在 __init__ 結束時是一個空字典。")
            else: self.logger.info(f"Stage2Dataset __init__ (mode={self.mode}) COMPLETED. 屬性 '{final_check_attr_name}' 已設定。長度: {len(attr_value)}")


    def __len__(self) -> int:
        return len(self.df_s2)

    def _calculate_stage2_target_flows(self) -> Dict[Tuple[int, int, int], np.ndarray]:
        self.logger.info(f"Stage2 (mode={self.mode}): 計算複合條件 (小時類別, 假日, 新條件類別) 的目標平均流量...")
        avg_flows: Dict[Tuple[int, int, int], np.ndarray] = {}

        if not hasattr(self, 'sorted_flow_columns') or not self.sorted_flow_columns or any(col == "" for col in self.sorted_flow_columns):
            self.logger.error(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - 'sorted_flow_columns' 缺失或無效。")
            return {}

        missing_cols = [col for col in self.sorted_flow_columns if col not in self.df_s2.columns]
        if missing_cols:
            self.logger.error(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - 以下流量欄位在 self.df_s2 中未找到: {missing_cols}")
            return {}

        if not (len(self.df_s2) == len(self.hour_category_for_target_grouping_np) == \
                len(self.is_holiday_for_target_np) == len(self.new_cond_category_for_target_np)): 
            self.logger.error(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - 用於分組的 Series 長度不一致。 "
                              f"df_s2: {len(self.df_s2)}, hour_cat: {len(self.hour_category_for_target_grouping_np)}, "
                              f"is_hol: {len(self.is_holiday_for_target_np)}, new_cond_cat: {len(self.new_cond_category_for_target_np)}")
            return {}

        try:
            flow_data_for_calc = self.df_s2[self.sorted_flow_columns].values.astype(np.float32)
        except KeyError as e:
            self.logger.error(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - 提取流量數據時發生 KeyError: {e}")
            return {}

        self.logger.info(f"_calc_target_flows (mode={self.mode}): df_s2 len: {len(self.df_s2)}")

        grouping_df = pd.DataFrame({
            'hour_category': self.hour_category_for_target_grouping_np,
            'is_holiday': self.is_holiday_for_target_np,
            'new_cond_category_for_target': self.new_cond_category_for_target_np 
        })

        if grouping_df.empty:
            self.logger.warning(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - Grouping DataFrame 為空。")
            return {}

        try:
            grouped = grouping_df.groupby(['hour_category', 'is_holiday', 'new_cond_category_for_target'], observed=False)
        except Exception as e:
            self.logger.error(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - Groupby 時發生錯誤: {e}")
            return {}

        if not grouped.groups or (grouped.groups and all(idx.empty for idx in grouped.groups.values())):
            self.logger.warning(f"Stage2Dataset (mode={self.mode}): _calculate_stage2_target_flows - 分組後 grouped.groups 為空或所有組都為空。")
            return {}
        else:
            self.logger.info(f"Number of groups found (mode={self.mode}): {len(grouped.groups)}")

        self.logger.info(f"Stage2 Target Calculation (mode={self.mode}): 樣本數分佈如下 (hour_category, is_holiday, new_cond_category): count") # 修改日誌
        for group_key, group_indices in grouped.indices.items():
            if len(group_indices) == 0:
                self.logger.debug(f"  - Group {group_key} is empty, skipping.")
                continue
            hr_cat, is_hol, new_cond_cat = group_key
            count = len(group_indices)
            self.logger.info(f"  - (hour_cat={hr_cat}, is_hol={is_hol}, new_cond_cat={new_cond_cat}): {count} samples") 

            group_flows_for_condition = flow_data_for_calc[group_indices]
            mean_flow_flat_for_condition = np.nanmean(group_flows_for_condition, axis=0)
            mean_flow_flat_for_condition[np.isnan(mean_flow_flat_for_condition)] = 0
            avg_flows[(hr_cat, int(is_hol), int(new_cond_cat))] = mean_flow_flat_for_condition.reshape(self.H, self.W) 

        if not avg_flows:
            self.logger.warning(f"Stage2 (mode={self.mode}): _calculate_stage2_target_flows - 計算完成，但 avg_flows 字典為空。")
            return {}

        self.logger.info(f"Stage2 (mode={self.mode}): 計算完成 {len(avg_flows)} 個 (小時類別,假日,新條件類別) 條件的目標平均流量圖。")
        return avg_flows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bm_output_grid_np_sample = self.basemodel_outputs_np[idx]
        if bm_output_grid_np_sample.shape[0] != 1:
            self.logger.warning(f"__getitem__ (idx {idx}): Basemodel output (condition 1) has {bm_output_grid_np_sample.shape[0]} channels, expected 1. Using first channel.")
            bm_output_grid_np_sample = bm_output_grid_np_sample[0:1, ...]
        condition_grid_1_tensor = torch.from_numpy(bm_output_grid_np_sample.astype(np.float32))

        if not hasattr(self, 'new_cond_feature_mean') or not hasattr(self, 'new_cond_feature_std'):
            self.logger.error(f"__getitem__ (idx {idx}, mode={self.mode}): new_cond_feature_mean 或 new_cond_feature_std 未在 __init__ 中設定!")
            fallback_mean, fallback_std = 0.0, 1.0
            original_value_for_cond2 = self.new_cond_original_values_np[idx]
            normalized_new_cond_value = (original_value_for_cond2 - fallback_mean) / fallback_std \
                if not np.isnan(original_value_for_cond2) else 0.0
        else:
            original_new_cond_value = self.new_cond_original_values_np[idx]
            if np.isnan(original_new_cond_value):
                normalized_new_cond_value = 0.0
            else:
                current_new_cond_std = self.new_cond_feature_std if self.new_cond_feature_std > 1e-6 else 1.0
                normalized_new_cond_value = (original_new_cond_value - self.new_cond_feature_mean) / current_new_cond_std
        
        condition_grid_2_tensor = torch.full(
            (1, self.D, self.H, self.W), float(normalized_new_cond_value), dtype=torch.float32
        )

        hr_original_for_basemodel = self.hours_for_target_np[idx]
        hr_cat_for_s2_target = self.hour_category_for_target_grouping_np[idx]
        is_hol = self.is_holiday_for_target_np[idx]
        new_cond_cat = self.new_cond_category_for_target_np[idx] 
        target_key = (hr_cat_for_s2_target, is_hol, new_cond_cat) 

        if not hasattr(self, 'average_flow_map_dict_s2') or not isinstance(self.average_flow_map_dict_s2, dict):
            self.logger.error(f"__getitem__ (idx {idx}, mode={self.mode}): self.average_flow_map_dict_s2 缺失或類型錯誤!")
            target_avg_flow_s2_np = np.zeros((self.H, self.W), dtype=np.float32)
        else:
            target_avg_flow_s2_np = self.average_flow_map_dict_s2.get(target_key)
            if target_avg_flow_s2_np is None:
                self.logger.debug(f"Stage2Dataset (idx {idx}, mode={self.mode}): 未找到目標鍵 {target_key}，使用零值網格。 "
                                 f"可用鍵數量: {len(self.average_flow_map_dict_s2)}")
                target_avg_flow_s2_np = np.zeros((self.H, self.W), dtype=np.float32)
        
        if hasattr(self, 'norm_stats_stage2_target') and self.norm_stats_stage2_target is not None:
            target_mean_to_use = self.norm_stats_stage2_target['mean']
            target_std_to_use = self.norm_stats_stage2_target['std']
            self.logger.debug(f"Using dedicated S2 target norm: mean={target_mean_to_use}, std={target_std_to_use}") # 可選日誌
        else:
            target_mean_to_use = self.config.get("cached_basemodel_mean", 0.0)
            target_std_to_use = self.config.get("cached_basemodel_std", 1.0)
            self.logger.debug(f"Using cached_basemodel_stats for S2 target norm: mean={target_mean_to_use}, std={target_std_to_use}") # 可選日誌

        norm_target_s2_np = (target_avg_flow_s2_np - target_mean_to_use) / target_std_to_use
        
        target_flow_tensor = torch.from_numpy(norm_target_s2_np).float().reshape(
            self.image_channels_target, self.D, self.H, self.W
        )
        
        original_hour_scalar_tensor = torch.tensor(hr_original_for_basemodel, dtype=torch.long)
        original_is_holiday_scalar_tensor = torch.tensor(is_hol, dtype=torch.long)
        
        return target_flow_tensor, condition_grid_1_tensor, condition_grid_2_tensor, \
               original_hour_scalar_tensor, original_is_holiday_scalar_tensor
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

    # 從 config 中獲取網格映射信息，這些信息應在加載 Basemodel 檢查點時被緩存
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

    eval_basemodel_mean = config.get("cached_basemodel_mean")
    eval_basemodel_std = config.get("cached_basemodel_std")

    if eval_basemodel_mean is None or eval_basemodel_std is None:
        logger.error(f"evaluate_stage2_models: CONFIG 中缺少 cached_basemodel_mean 或 cached_basemodel_std。無法進行反正規化。")
        # 返回一個表示錯誤的結構
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape": float('nan'), "smape": float('nan'), "fid": float('nan')}
        nan_grids_dict = {'MSE': np.array([np.nan]), 'MAE': np.array([np.nan]), 'MAPE': np.array([np.nan]), 'SMAPE': np.array([np.nan])} # 提供一個預設的 ndarray
        return {"stage2_model": nan_metrics, "basemodel_on_s2_data": nan_metrics}, \
               {"stage2_model": nan_grids_dict, "basemodel_on_s2_data": nan_grids_dict}

    if eval_basemodel_std < 1e-6:
        logger.warning(f"evaluate_stage2_models: cached_basemodel_std ({eval_basemodel_std}) 過小，將其視為 1.0 進行反正規化。")
        eval_basemodel_std = 1.0

    all_s2_generated_denorm_list: List[torch.Tensor] = []
    all_bm_generated_denorm_on_s2_data_list: List[torch.Tensor] = []
    all_s2_target_denorm_list: List[torch.Tensor] = []

    all_s2_generated_norm_for_fid_list: List[torch.Tensor] = []
    all_bm_generated_norm_for_fid_on_s2_data_list: List[torch.Tensor] = []
    all_s2_target_norm_for_fid_list: List[torch.Tensor] = []

    # --- 從 dataloader_s2 獲取 dataset 物件 ---
    if not hasattr(dataloader_s2, 'dataset') or dataloader_s2.dataset is None:
        logger.error("evaluate_stage2_models: dataloader_s2 沒有 dataset 屬性或 dataset 為 None！")
        # 返回錯誤結構
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape": float('nan'), "smape": float('nan'), "fid": float('nan')}
        nan_grids_dict = {'MSE': np.array([np.nan]), 'MAE': np.array([np.nan]), 'MAPE': np.array([np.nan]), 'SMAPE': np.array([np.nan])}
        return {"stage2_model": nan_metrics, "basemodel_on_s2_data": nan_metrics}, \
               {"stage2_model": nan_grids_dict, "basemodel_on_s2_data": nan_grids_dict}
    
    dataset_s2_obj_for_eval = dataloader_s2.dataset 
    if hasattr(dataset_s2_obj_for_eval, 'norm_stats_stage2_target') and \
       dataset_s2_obj_for_eval.norm_stats_stage2_target is not None:
        # 確保從 dataset 物件中獲取 (因為 test_dataset 在初始化時已經接收了這些統計量)
        current_s2_target_stats = dataset_s2_obj_for_eval.norm_stats_stage2_target
        target_mean_for_denorm = current_s2_target_stats['mean']
        target_std_for_denorm = current_s2_target_stats['std']
        logger.info(f"evaluate_stage2_models: 使用 Stage2 目標專用統計量進行反正規化: mean={target_mean_for_denorm:.4f}, std={target_std_for_denorm:.4f}")
    else: # 回退到 cached_basemodel_stats
        target_mean_for_denorm = config.get("cached_basemodel_mean")
        target_std_for_denorm = config.get("cached_basemodel_std")
        logger.info(f"evaluate_stage2_models: 使用 cached_basemodel_stats 進行反正規化: mean={target_mean_for_denorm:.4f}, std={target_std_for_denorm:.4f}")

    if target_mean_for_denorm is None or target_std_for_denorm is None:
        logger.error(f"evaluate_stage2_models: 無法獲取用於反正規化的均值或標準差。")
        # 返回錯誤結構
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape": float('nan'), "smape": float('nan'), "fid": float('nan')}
        nan_grids_dict = {'MSE': np.array([np.nan]), 'MAE': np.array([np.nan]), 'MAPE': np.array([np.nan]), 'SMAPE': np.array([np.nan])}
        return {"stage2_model": nan_metrics, "basemodel_on_s2_data": nan_metrics}, \
               {"stage2_model": nan_grids_dict, "basemodel_on_s2_data": nan_grids_dict}
    
    if target_std_for_denorm < 1e-6:
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

        all_s2_generated_denorm_list.append(s2_generated_eval_denorm.cpu())
        all_bm_generated_denorm_on_s2_data_list.append(bm_generated_denorm_on_s2_conditions.cpu())
        all_s2_target_denorm_list.append(s2_target_eval_denorm.cpu())

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
        nan_grids_dict = {'MSE': np.array([np.nan]), 'MAE': np.array([np.nan]), 'MAPE': np.array([np.nan]), 'SMAPE': np.array([np.nan])}
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

        smape_num = torch.abs(pred_t - s2_target_all_t)
        smape_den = (torch.abs(s2_target_all_t) + torch.abs(pred_t)) / 2.0 + epsilon
        smape_tensor = (smape_num / smape_den) * 100
        smape = torch.mean(smape_tensor[torch.isfinite(smape_tensor)]).item() if torch.isfinite(smape_tensor).any() else float('inf')

        fid = float('nan')
        current_generated_norm_for_fid_list_to_use: List[torch.Tensor] = []
        if model_name == "stage2_model":
            current_generated_norm_for_fid_list_to_use = all_s2_generated_norm_for_fid_list
        elif model_name == "basemodel_on_s2_data":
            current_generated_norm_for_fid_list_to_use = all_bm_generated_norm_for_fid_on_s2_data_list

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
            original_all_denorm_t=torch.mean(s2_target_all_t, dim=0, keepdim=True).clone().cpu(),    # 平均目標
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

    # 繪製誤差地理圖
    if "stage2_model" in error_grids_all_models and isinstance(error_grids_all_models["stage2_model"], dict):
        plot_grid_with_error_long_term(
            dataset_s2_obj_for_eval, # 傳遞 Dataset 物件
            error_grids_all_models["stage2_model"],
            config,
            f"{prefix}_stage2"
        )
    if "basemodel_on_s2_data" in error_grids_all_models and isinstance(error_grids_all_models["basemodel_on_s2_data"], dict):
        plot_grid_with_error_long_term(
            dataset_s2_obj_for_eval, # 傳遞 Dataset 物件
            error_grids_all_models["basemodel_on_s2_data"],
            config,
            f"{prefix}_basemodel"
        )

    # 計算並繪製誤差差異圖
    if "stage2_model" in error_grids_all_models and isinstance(error_grids_all_models["stage2_model"], dict) and \
       "basemodel_on_s2_data" in error_grids_all_models and isinstance(error_grids_all_models["basemodel_on_s2_data"], dict):
        s2_errors = error_grids_all_models["stage2_model"]
        bm_errors = error_grids_all_models["basemodel_on_s2_data"]
        error_metrics_difference_grids = {}

        for metric_key in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
            if metric_key in s2_errors and isinstance(s2_errors[metric_key], np.ndarray) and \
               metric_key in bm_errors and isinstance(bm_errors[metric_key], np.ndarray) and \
               s2_errors[metric_key].shape == bm_errors[metric_key].shape:
                difference_grid = s2_errors[metric_key] - bm_errors[metric_key]
                error_metrics_difference_grids[f"Diff_{metric_key}_(S2-BM)"] = difference_grid
            else:
                logger.warning(f"無法計算指標 '{metric_key}' 的差異網格，因數據缺失、類型錯誤或形狀不匹配。")
        
        if error_metrics_difference_grids:
            plot_grid_with_error_long_term(
                dataset_s2_obj_for_eval, # 傳遞 Dataset 物件
                error_metrics_difference_grids,
                config,
                f"{prefix}_diff_S2_minus_BM"
            )

    return results, error_grids_all_models


if __name__ == '__main__':
    logger.info(f"===== DDPM Stage 2 Training and Evaluation =====")
    logger.info(f"Full CONFIG: {json.dumps(CONFIG, indent=2)}")

    # --- 載入完整數據 ---
    full_df = pd.read_csv(CONFIG["data_path"])
    logger.info(f"已載入資料: {CONFIG['data_path']}. 形狀: {full_df.shape}")

    # === 步驟 1: 載入預訓練的 Basemodel (僅用於生成條件輸入) ===
    # basemodel_for_output_generation 實例將使用其原始的 DDPM3D.sample 邏輯
    # (即接收小時和假日純量，內部轉換為網格)
    BASEMODEL_CHECKPOINT_PATH = CONFIG["basemodel_checkpoint_to_load_for_stage2"]
    if not os.path.exists(BASEMODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"未找到 Basemodel 檢查點: {BASEMODEL_CHECKPOINT_PATH}")

    logger.info(f"===== 載入 Basemodel (for output generation) 從: {BASEMODEL_CHECKPOINT_PATH} =====")
    chkpt_basemodel_eval = torch.load(BASEMODEL_CHECKPOINT_PATH, map_location=CONFIG["device"], weights_only=False)
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

# --- 步驟 2: 準備 Stage2 數據 ---
NEW_COND_FEATURE_COL = CONFIG["stage2_new_condition_feature_column"]
NEW_COND_OPERATOR = CONFIG["stage2_new_conditional_operator"]
NEW_COND_VALUE = CONFIG["stage2_new_conditional_value"]
STAGE2_MODEL_NAME = CONFIG["stage2_model_name"]


logger.info(f"===== STAGE 2: 數據準備 =====")
logger.info(f"Stage2 模型將學習處理基於 '{NEW_COND_FEATURE_COL} {NEW_COND_OPERATOR} {NEW_COND_VALUE}' 條件劃分的兩個數據分支。")

df_for_stage2_processing = full_df.copy() # 直接使用完整的 DataFrame

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
logger.info(f"Stage2: Basemodel 輸出 (條件) 正規化完畢, 形狀: {all_bm_outputs_s2_np_cond_normalized.shape}")
logger.info(f"正規化後 Basemodel 輸出的均值: {np.mean(all_bm_outputs_s2_np_cond_normalized):.4f}, 標準差: {np.std(all_bm_outputs_s2_np_cond_normalized):.4f}")

# --- 步驟 3: 初始化 Stage2 模型並從 Basemodel 檢查點遷移權重 ---
logger.info(f"===== STAGE 2: 初始化模型 '{STAGE2_MODEL_NAME}' 並從 Basemodel 遷移權重 =====")
config_for_stage2_model_creation = CONFIG.copy()
config_for_stage2_model_creation["condition_input_channels"] = CONFIG.get("stage2_ddpm_condition_input_channels", 2) # 確保為2

stage2_model = create_stage2_model_from_basemodel_checkpoint(
    basemodel_checkpoint_path=CONFIG['basemodel_checkpoint_to_load_for_stage2'],
    config_for_stage2_model=config_for_stage2_model_creation,
    device=CONFIG["device"]
)

# --- 步驟 4: 準備 Stage2 的 Dataset 和 DataLoader ---
s2_indices_all = np.arange(len(df_for_stage2_processing))
np.random.shuffle(s2_indices_all) # 使用全局種子
s2_train_len_final = int(CONFIG["train_split_ratio"] * len(s2_indices_all))
s2_val_len_final = int(CONFIG["val_split_ratio"] * len(s2_indices_all))

s2_train_indices_final = s2_indices_all[:s2_train_len_final]
s2_val_indices_final = s2_indices_all[s2_train_len_final : s2_train_len_final + s2_val_len_final]
s2_test_indices_final = s2_indices_all[s2_train_len_final + s2_val_len_final:]

logger.info(f"Stage2 資料分割 (基於篩選後數據): 訓練集={len(s2_train_indices_final)}, 驗證集={len(s2_val_indices_final)}, 測試集={len(s2_test_indices_final)}")

config_for_s2_dataset_use = CONFIG.copy()

train_dataset_s2 = Stage2Dataset(
    df_for_stage2_processing=df_for_stage2_processing.iloc[s2_train_indices_final],
    basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_train_indices_final],
    config=config_for_s2_dataset_use, # <--- 這個 config 傳遞給 Dataset
    mode='train',
    original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
    # new_cond_feature_norm_stats_from_train 在訓練模式下由 Dataset 內部計算
)
s2_train_batch_size = CONFIG.get("batch_size")
train_loader_s2 = DataLoader(train_dataset_s2, batch_size=s2_train_batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True if len(train_dataset_s2) >= s2_train_batch_size else False)

val_loader_s2 = None
if len(s2_val_indices_final) > 0:
    logger.info("Attempting to create val_dataset_s2...")
    logger.info(f"BEFORE val_dataset_s2 creation: train_dataset_s2.average_flow_map_dict_s2 is None: {train_dataset_s2.average_flow_map_dict_s2 is None}")
    if train_dataset_s2.average_flow_map_dict_s2 is not None:
        logger.info(f"Length of train_dataset_s2.average_flow_map_dict_s2 before val_dataset_s2: {len(train_dataset_s2.average_flow_map_dict_s2)}")
    else:
        logger.error("CRITICAL: train_dataset_s2.average_flow_map_dict_s2 IS NONE BEFORE val_dataset_s2 creation!")

    s2_target_stats_for_val_test = None
    if hasattr(train_dataset_s2, 'norm_stats_stage2_target') and \
       train_dataset_s2.norm_stats_stage2_target is not None:
        s2_target_stats_for_val_test = train_dataset_s2.norm_stats_stage2_target
    else:
        logger.warning("無法從 train_dataset_s2 獲取 norm_stats_stage2_target 傳遞給 val_dataset_s2。這可能導致錯誤或使用預設值。")

    val_dataset_s2 = Stage2Dataset(
        df_for_stage2_processing=df_for_stage2_processing.iloc[s2_val_indices_final],
        basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_val_indices_final],
        config=config_for_s2_dataset_use,
        mode='val',
        stage2_avg_flow_map_dict_from_train=train_dataset_s2.average_flow_map_dict_s2,
        original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
        new_cond_feature_norm_stats_from_train=train_dataset_s2.norm_stats_new_cond_feature,
        stage2_target_norm_stats_from_train=s2_target_stats_for_val_test
    )
    s2_eval_batch_size = CONFIG.get("eval_batch_size") 
    val_loader_s2 = DataLoader(val_dataset_s2, batch_size=s2_eval_batch_size, shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    logger.info(f"Stage2 驗證數據集創建，含 {len(val_dataset_s2)} 樣本。")
else:
    logger.info("Stage2 驗證集為空。")

# --- 步驟 5: 訓練 Stage2 模型 ---
# --- 初始化 optimizer, scheduler, 狀態變數 ---
optimizer_s2 = optim.AdamW(list(stage2_model.parameters()), lr=CONFIG.get("lr_stage2", CONFIG.get("lr", 1e-3)), weight_decay=CONFIG.get("weight_decay", 1e-5))
scheduler_factor_s2 = CONFIG.get("lr_scheduler_factor", 0.5)
scheduler_patience_s2 = CONFIG.get("lr_scheduler_patience", 3)
scheduler_min_lr_s2 = CONFIG.get("lr_scheduler_min_lr", 1e-6)
early_stopping_patience_s2 = CONFIG.get("early_stopping_patience", 6)

scheduler_s2 = ReduceLROnPlateau(optimizer_s2,
                                 mode='min',
                                 factor=scheduler_factor_s2,
                                 patience=scheduler_patience_s2,
                                 min_lr=scheduler_min_lr_s2)

start_epoch_s2_train = 1
best_val_loss_s2_train = float('inf')
early_stopping_counter_s2_train = 0
stage2_model_save_checkpoint_path_full = CONFIG["stage2_checkpoint_full_path"]
metrics_hist_s2_train = {'train_loss':[], 'val_loss':[], 'lr':[]}

# --- 從檢查點恢復 (如果設定) ---
if CONFIG.get("resume_from_stage2_checkpoint", True) and os.path.exists(stage2_model_save_checkpoint_path_full):
    logger.info(f"從 Stage2 檢查點恢復訓練: {stage2_model_save_checkpoint_path_full}")
    try:
        chkpt_s2_resume = torch.load(stage2_model_save_checkpoint_path_full, map_location=CONFIG["device"], weights_only=False)
        stage2_model.load_state_dict(chkpt_s2_resume['ddpm_state_dict'])
        optimizer_s2.load_state_dict(chkpt_s2_resume['optimizer_state_dict'])
        if 'scheduler_state_dict' in chkpt_s2_resume and chkpt_s2_resume['scheduler_state_dict']:
            scheduler_s2.load_state_dict(chkpt_s2_resume['scheduler_state_dict'])
        start_epoch_s2_train = chkpt_s2_resume.get('epoch', 0) + 1
        best_val_loss_s2_train = chkpt_s2_resume.get('best_val_loss_s2', float('inf'))
        early_stopping_counter_s2_train = chkpt_s2_resume.get('early_stopping_counter_s2',0)
        metrics_hist_s2_train = chkpt_s2_resume.get('metrics_hist_s2', {'train_loss':[], 'val_loss':[], 'lr':[]})
        resumed_stage2_target_stats = chkpt_s2_resume.get('norm_stats_stage2_target')
        if resumed_stage2_target_stats is not None:
            CONFIG["cached_stage2_target_mean"] = resumed_stage2_target_stats.get('mean')
            CONFIG["cached_stage2_target_std"] = resumed_stage2_target_stats.get('std')
            logger.info(f"從檢查點恢復 Stage2 目標專用正規化統計量: mean={CONFIG['cached_stage2_target_mean']}, std={CONFIG['cached_stage2_target_std']}")
        else:
            logger.warning("檢查點中未找到 'norm_stats_stage2_target'，如果啟用專用統計量，Dataset 初始化可能會出錯或使用預設值。")
        logger.info(f"Stage2 訓練將從 epoch {start_epoch_s2_train} 開始。最佳驗證損失: {best_val_loss_s2_train:.5f}")
    except Exception as e:
        logger.error(f"從檢查點恢復訓練失敗: {e}。將從頭開始訓練。")
        start_epoch_s2_train = 1
        best_val_loss_s2_train = float('inf')
        early_stopping_counter_s2_train = 0
        metrics_hist_s2_train = {'train_loss':[], 'val_loss':[], 'lr':[]}

epochs_to_run_s2 = CONFIG.get("epochs_stage2", CONFIG.get("epochs", 100))
logger.info(f"開始訓練 Stage2 模型: {STAGE2_MODEL_NAME} for {epochs_to_run_s2} epochs...")

# --- 主 Epoch 迴圈 ---
epoch_pbar = tqdm(range(start_epoch_s2_train, epochs_to_run_s2 + 1),
                    desc=f"Stage2 Training ({STAGE2_MODEL_NAME})",
                    leave=True, # 完成後保留最後狀態
                    position=0,
                    dynamic_ncols=True, # 允許進度條適應終端寬度
                    unit="epoch"
                    )

for epoch_s2_current in epoch_pbar:
    stage2_model.train()
    total_train_loss_epoch_s2 = 0.0

    # 內部訓練批次迴圈的進度條
    train_pbar_s2_loop = tqdm(train_loader_s2,
                              desc=f"Epoch {epoch_s2_current} [Train]",
                              leave=False, # 完成後清除此內部進度條
                              position=1,  # 顯示在主進度條下方
                              dynamic_ncols=True,
                              unit="batch")

    for target_s2_b, bm_out_grid_b, new_cond_grid_b, _, _ in train_pbar_s2_loop:
        optimizer_s2.zero_grad()
        target_s2_b = target_s2_b.to(CONFIG["device"])
        bm_out_grid_b = bm_out_grid_b.to(CONFIG["device"])
        new_cond_grid_b = new_cond_grid_b.to(CONFIG["device"])

        t_s2_b = torch.randint(0, stage2_model.timesteps, (target_s2_b.shape[0],), device=CONFIG["device"]).long()
        loss_s2_batch = stage2_model.p_losses(
            x_start_target_flow=target_s2_b,
            t=t_s2_b,
            basemodel_output_grid_batch=bm_out_grid_b,
            new_condition_feature_grid_batch=new_cond_grid_b
        )
        loss_s2_batch.backward()
        optimizer_s2.step()
        total_train_loss_epoch_s2 += loss_s2_batch.item()
        train_pbar_s2_loop.set_postfix({"Batch Loss": f"{loss_s2_batch.item():.5f}"})

    avg_train_loss_epoch_s2 = total_train_loss_epoch_s2 / len(train_loader_s2) if len(train_loader_s2) > 0 else 0.0
    metrics_hist_s2_train['train_loss'].append(avg_train_loss_epoch_s2)
    current_lr_epoch_s2 = optimizer_s2.param_groups[0]['lr']
    metrics_hist_s2_train['lr'].append(current_lr_epoch_s2)

    # --- 驗證邏輯 ---
    avg_val_loss_s2_to_record = float('inf')
    val_calculated_this_epoch = False
    val_freq_s2 = CONFIG.get("val_calculation_freq_stage2", CONFIG.get("val_calculation_freq", 1))

    should_validate_this_epoch = False
    if val_loader_s2:
        if epoch_s2_current == epochs_to_run_s2:
            should_validate_this_epoch = True
        elif epoch_s2_current > start_epoch_s2_train and (epoch_s2_current - start_epoch_s2_train) % val_freq_s2 == 0:
            should_validate_this_epoch = True

    if should_validate_this_epoch:
        val_calculated_this_epoch = True
        stage2_model.eval()
        total_val_loss_p_s2_epoch = 0.0
        num_val_samples_p_s2_epoch = 0
        actual_avg_val_loss_this_epoch = float('inf')

        if hasattr(val_loader_s2, 'dataset') and len(val_loader_s2.dataset) > 0:
            # 計算 Val Loss 時顯示進度條
            val_pbar_s2_loop = tqdm(val_loader_s2,
                                    desc=f"Epoch {epoch_s2_current} [S2 Validate]",
                                    leave=False, # 完成後清除
                                    position=1,  # 顯示在主進度條下方
                                    dynamic_ncols=True,
                                    unit="batch")
            with torch.no_grad():
                for target_s2_val_norm, bm_out_val_cond, new_cond_val_cond, \
                    _, _ in val_pbar_s2_loop: # val_pbar_s2_loop 應該會顯示進度
                    target_s2_val_norm = target_s2_val_norm.to(CONFIG["device"])
                    bm_out_val_cond = bm_out_val_cond.to(CONFIG["device"])
                    new_cond_val_cond = new_cond_val_cond.to(CONFIG["device"])

                    s2_generated_val_norm = stage2_model.sample(
                        batch_size=target_s2_val_norm.shape[0],
                        basemodel_output_grid_batch=bm_out_val_cond,
                        new_condition_feature_grid_batch=new_cond_val_cond
                    )
                    val_target_mean_to_use: float
                    val_target_std_to_use: float
                    if hasattr(train_dataset_s2, 'norm_stats_stage2_target') and \
                       train_dataset_s2.norm_stats_stage2_target is not None: # 驗證時直接從 train_dataset_s2 取
                        
                        current_s2_target_stats_val = train_dataset_s2.norm_stats_stage2_target
                        val_target_mean_to_use = current_s2_target_stats_val['mean']
                        val_target_std_to_use = current_s2_target_stats_val['std']
                        if val_target_std_to_use < 1e-6: val_target_std_to_use = 1.0
                    else: # 回退到 cached_basemodel_stats
                        val_target_mean_to_use = CONFIG.get("cached_basemodel_mean")
                        val_target_std_to_use = CONFIG.get("cached_basemodel_std")
                        if val_target_mean_to_use is None or val_target_std_to_use is None:
                            if epoch_s2_current == start_epoch_s2_train or not metrics_hist_s2_train['val_loss'] or metrics_hist_s2_train['val_loss'][-1] == float('inf') :
                                tqdm.write(f"ERROR: Epoch {epoch_s2_current}: CONFIG 中缺少 Base Model 正規化統計量，無法計算反正規化 Val Loss。")
                            val_loss_b_s2 = float('nan')
                            continue 
                        if val_target_std_to_use < 1e-6: val_target_std_to_use = 1.0
                    
                    s2_generated_val_denorm = s2_generated_val_norm * val_target_std_to_use + val_target_mean_to_use
                    s2_target_val_denorm = target_s2_val_norm * val_target_std_to_use + val_target_mean_to_use
                    s2_generated_val_denorm = torch.clamp(s2_generated_val_denorm, min=0.0)

                    val_loss_b_s2 = F.mse_loss(s2_generated_val_denorm, s2_target_val_denorm).item()

                    if not np.isnan(val_loss_b_s2):
                        total_val_loss_p_s2_epoch += val_loss_b_s2 * target_s2_val_norm.shape[0]
                        num_val_samples_p_s2_epoch += target_s2_val_norm.shape[0]
                    val_pbar_s2_loop.set_postfix({"Val Batch MSE": f"{val_loss_b_s2:.5f}" if not np.isnan(val_loss_b_s2) else "NaN"})
            
            if num_val_samples_p_s2_epoch > 0:
                actual_avg_val_loss_this_epoch = total_val_loss_p_s2_epoch / num_val_samples_p_s2_epoch
            else: # 驗證集dataset非空，但遍歷後num_val_samples_p_s2_epoch為0（例如batch size > dataset size）
                logger.warning(f"Stage2 Epoch {epoch_s2_current}: 驗證時處理的樣本數為0，無法計算有效驗證損失。")
                actual_avg_val_loss_this_epoch = float('inf')
        else: # val_loader_s2.dataset 為空
            logger.info(f"Stage2 Epoch {epoch_s2_current}: 驗證數據加載器為空或其數據集為空，跳過驗證損失計算。")
            actual_avg_val_loss_this_epoch = float('inf')


        avg_val_loss_s2_to_record = actual_avg_val_loss_this_epoch

        if actual_avg_val_loss_this_epoch != float('inf'):
            scheduler_s2.step(actual_avg_val_loss_this_epoch)
            if actual_avg_val_loss_this_epoch < best_val_loss_s2_train:
                best_val_loss_s2_train = actual_avg_val_loss_this_epoch
                early_stopping_counter_s2_train = 0
                tqdm.write(f"Epoch {epoch_s2_current}: 新最佳模型已儲存 (Val Loss: {best_val_loss_s2_train:.5f})。") # 使用 tqdm.write
                avg_flow_map_to_save = train_dataset_s2.average_flow_map_dict_s2 if 'train_dataset_s2' in locals() and hasattr(train_dataset_s2, 'average_flow_map_dict_s2') else None
                new_cond_stats_to_save = train_dataset_s2.norm_stats_new_cond_feature if 'train_dataset_s2' in locals() and hasattr(train_dataset_s2, 'norm_stats_new_cond_feature') else None
                if avg_flow_map_to_save is None or new_cond_stats_to_save is None :
                     tqdm.write(f"WARNING: Epoch {epoch_s2_current}: 無法獲取 train_dataset_s2 的統計數據以儲存到檢查點。")

                torch.save({
                    'epoch': epoch_s2_current,
                    'ddpm_state_dict': stage2_model.state_dict(),
                    'optimizer_state_dict': optimizer_s2.state_dict(),
                    'scheduler_state_dict': scheduler_s2.state_dict(),
                    'best_val_loss_s2': best_val_loss_s2_train,
                    'config_snapshot_at_save': config_for_s2_dataset_use, # 使用傳給Dataset的config
                    'metrics_hist_s2': metrics_hist_s2_train,
                    'early_stopping_counter_s2': early_stopping_counter_s2_train,
                    'stage2_avg_flow_map_dict': avg_flow_map_to_save,
                    'new_cond_feature_norm_stats': new_cond_stats_to_save,
                    'norm_stats_stage2_target': train_dataset_s2.norm_stats_stage2_target
                }, stage2_model_save_checkpoint_path_full)
            else: # 驗證損失沒有改善
                early_stopping_counter_s2_train += 1
    else: # 本 epoch 不執行驗證
        avg_val_loss_s2_to_record = metrics_hist_s2_train['val_loss'][-1] if metrics_hist_s2_train['val_loss'] else float('inf')

    metrics_hist_s2_train['val_loss'].append(avg_val_loss_s2_to_record)
    val_loss_display_s2 = f"{avg_val_loss_s2_to_record:.5f}" if avg_val_loss_s2_to_record != float('inf') else "N/A"
    if val_calculated_this_epoch and avg_val_loss_s2_to_record != float('inf'):
        val_loss_display_s2 += " (Calc)"

    # 更新主 epoch 進度條的後綴信息
    epoch_pbar.set_postfix_str(f"Tr_Loss: {avg_train_loss_epoch_s2:.4f}, Val_Loss: {val_loss_display_s2}, LR: {current_lr_epoch_s2:.1e}, ES: {early_stopping_counter_s2_train}/{early_stopping_patience_s2}")

    if early_stopping_counter_s2_train >= early_stopping_patience_s2:
        tqdm.write(f"Stage2 訓練因早停機制觸發於 Epoch {epoch_s2_current} (計數器: {early_stopping_counter_s2_train})。")
        break # 跳出 epoch 迴圈

# 確保在迴圈結束後關閉主進度條
if 'epoch_pbar' in locals() and isinstance(epoch_pbar, tqdm):
    epoch_pbar.close()

logger.info(f"Stage2 模型 '{STAGE2_MODEL_NAME}' 訓練完成。")
# 訓練完成後，可以選擇打印一次最終的統計數據（這部分 logger.info 可以保留）
final_train_loss = metrics_hist_s2_train['train_loss'][-1] if metrics_hist_s2_train['train_loss'] else float('nan')
final_val_loss = metrics_hist_s2_train['val_loss'][-1] if metrics_hist_s2_train['val_loss'] else float('nan')
final_lr = metrics_hist_s2_train['lr'][-1] if metrics_hist_s2_train['lr'] else float('nan')
logger.info(f"最終訓練統計: Train Loss: {final_train_loss:.5f}, Last Recorded Val Loss: {final_val_loss:.5f}, Final LR: {final_lr:.8f}")
if best_val_loss_s2_train != float('inf'):
    logger.info(f"最佳驗證損失記錄: {best_val_loss_s2_train:.5f}")

# --- Stage2 模型最終評估 ---
logger.info(f"===== STAGE 2: 最終模型評估 ({STAGE2_MODEL_NAME}) =====")
if not os.path.exists(stage2_model_save_checkpoint_path_full):
    logger.warning(f"找不到最佳 Stage2 模型檔案: {stage2_model_save_checkpoint_path_full}。將使用訓練結束時的 Stage2 模型狀態進行評估。")
    final_s2_model_for_eval_load = stage2_model
    chkpt_s2_final_for_eval = {'epoch': epochs_to_run_s2 } # 模擬一個檢查點字典
    # 嘗試從 train_dataset_s2 獲取統計數據，如果模型是訓練結束時的狀態
    s2_target_stats_for_final_eval = train_dataset_s2.norm_stats_target_s2 if hasattr(train_dataset_s2, 'norm_stats_target_s2') else None
    s2_avg_flow_map_for_final_eval = train_dataset_s2.average_flow_map_dict_s2 if hasattr(train_dataset_s2, 'average_flow_map_dict_s2') else None
    new_cond_feature_norm_stats_for_final_eval = train_dataset_s2.norm_stats_new_cond_feature if hasattr(train_dataset_s2, 'norm_stats_new_cond_feature') else None
    stage2_target_norm_stats_for_final_eval = train_dataset_s2.norm_stats_stage2_target if hasattr(train_dataset_s2, 'norm_stats_stage2_target') else None
else:
    logger.info(f"從 {stage2_model_save_checkpoint_path_full} 載入最佳 Stage2 模型進行評估...")
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
        eval_s2_unet_final,
        config_from_s2_chkpt_eval.get("timesteps", CONFIG["timesteps"]),
        (config_from_s2_chkpt_eval.get("D", CONFIG["D"]), config_from_s2_chkpt_eval.get("H", CONFIG["H"]), config_from_s2_chkpt_eval.get("W", CONFIG["W"])),
        config_from_s2_chkpt_eval.get("image_channels", CONFIG["image_channels"]),
        config_from_s2_chkpt_eval.get("stage2_ddpm_condition_input_channels", CONFIG.get("stage2_ddpm_condition_input_channels", 2)),
        config_from_s2_chkpt_eval.get("condition_encode_dim", CONFIG["condition_encode_dim"]),
        beta_start=config_from_s2_chkpt_eval.get("beta_start", CONFIG["beta_start"]),
        beta_end=config_from_s2_chkpt_eval.get("beta_end", CONFIG["beta_end"]),
        device=CONFIG["device"]
    )
    final_s2_model_to_eval.load_state_dict(chkpt_s2_final_for_eval['ddpm_state_dict'])
    logger.info(f"最佳 Stage2 模型 (Epoch {chkpt_s2_final_for_eval.get('epoch','未知')}) 載入完成。")
    s2_avg_flow_map_for_final_eval = chkpt_s2_final_for_eval.get('stage2_avg_flow_map_dict')
    new_cond_feature_norm_stats_for_final_eval = chkpt_s2_final_for_eval.get('new_cond_feature_norm_stats')
    stage2_target_norm_stats_for_final_eval = train_dataset_s2.norm_stats_stage2_target if hasattr(train_dataset_s2, 'norm_stats_stage2_target') else None

if s2_avg_flow_map_for_final_eval is None :
    if hasattr(train_dataset_s2, 'average_flow_map_dict_s2'): s2_avg_flow_map_for_final_eval = train_dataset_s2.average_flow_map_dict_s2
if new_cond_feature_norm_stats_for_final_eval is None :
    if hasattr(train_dataset_s2, 'norm_stats_new_cond_feature'): new_cond_feature_norm_stats_for_final_eval = train_dataset_s2.norm_stats_new_cond_feature
if CONFIG.get("use_dedicated_stage2_target_norm", False) and stage2_target_norm_stats_for_final_eval is None:
    if hasattr(train_dataset_s2, 'norm_stats_stage2_target'): stage2_target_norm_stats_for_final_eval = train_dataset_s2.norm_stats_stage2_target

if s2_avg_flow_map_for_final_eval is None or new_cond_feature_norm_stats_for_final_eval is None or \
   (CONFIG.get("use_dedicated_stage2_target_norm", False) and stage2_target_norm_stats_for_final_eval is None):
     raise ValueError("無法為最終評估獲取必要的統計量 (avg_flow_map, new_cond_norm_stats, 或 stage2_target_norm_stats)。")

# 準備測試集 Loader
test_loader_s2_final = None
if len(s2_test_indices_final) > 0:
    test_dataset_s2_final = Stage2Dataset(
        df_for_stage2_processing=df_for_stage2_processing.iloc[s2_test_indices_final],
        basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_test_indices_final],
        config=config_for_s2_dataset_use,
        mode='test',
        stage2_avg_flow_map_dict_from_train=s2_avg_flow_map_for_final_eval, 
        original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
        new_cond_feature_norm_stats_from_train=new_cond_feature_norm_stats_for_final_eval, 
        stage2_target_norm_stats_from_train=stage2_target_norm_stats_for_final_eval if CONFIG.get("use_dedicated_stage2_target_norm", False) else None
    )
    s2_eval_batch_size_final = CONFIG.get("eval_batch_size")
    test_loader_s2_final = DataLoader(test_dataset_s2_final, batch_size=s2_eval_batch_size_final, shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    logger.info(f"Stage2 最終評估測試數據集創建，含 {len(test_dataset_s2_final)} 樣本。")

if test_loader_s2_final and len(test_loader_s2_final.dataset) > 0 :
    # 載入 Inception 模型
    inception_fid_eval = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True).to(CONFIG["device"])
    inception_fid_eval.fc = nn.Identity()
    if hasattr(inception_fid_eval, 'AuxLogits') and inception_fid_eval.AuxLogits is not None: inception_fid_eval.AuxLogits = None
    inception_fid_eval.eval()

    s2_final_eval_results, s2_final_error_grids = evaluate_stage2_models(
        stage2_model_trained=final_s2_model_to_eval,
        basemodel_eval_instance=basemodel_for_output_generation,
        dataloader_s2=test_loader_s2_final,
        inception_model_fid=inception_fid_eval,
        config=CONFIG,
        max_samples_for_fid=CONFIG.get("fid_num_samples_stage2"),
        prefix=f"final_eval_{STAGE2_MODEL_NAME}"
    )
    logger.info(f"--- Stage2 模型 ({STAGE2_MODEL_NAME}) 最終評估結果 (測試集) ---")
    if "stage2_model" in s2_final_eval_results: logger.info(f"Stage2 Model: {s2_final_eval_results['stage2_model']}")
    if "basemodel_on_s2_data" in s2_final_eval_results: logger.info(f"Basemodel (on S2 data): {s2_final_eval_results['basemodel_on_s2_data']}")
    
    # 保存指標到 JSON
    eval_metrics_path = os.path.join(CONFIG["stage2_model_save_dir"], f"final_evaluation_metrics_{STAGE2_MODEL_NAME}.json")
    with open(eval_metrics_path, 'w') as f: json.dump(s2_final_eval_results, f, indent=4)
    logger.info(f"Stage2 評估指標已儲存至: {eval_metrics_path}")
    
    # Excel 輸出
    excel_rows_final_s2 = []
    num_grid_cells_final = CONFIG["H"] * CONFIG["W"]
    grid_idx_to_rc_map_s2 = CONFIG.get("cached_basemodel_grid_idx_to_rc_map")
    sorted_flow_columns_s2 = CONFIG.get("cached_basemodel_sorted_flow_columns")
    selected_sensor_info_s2 = CONFIG.get("cached_basemodel_selected_sensor_info")
    if not grid_idx_to_rc_map_s2 or not sorted_flow_columns_s2 or not selected_sensor_info_s2:
        logger.error("無法獲取用於 Excel 報告的網格映射資訊。座標將不正確。")
        # 可以選擇在這裡返回或用預設值填充

    sensor_info_lookup_s2 = {info['name']: {'lon': info['lon'], 'lat': info['lat']}
                            for info in selected_sensor_info_s2 if isinstance(info, dict) and 'name' in info}
    for model_key_eval in ["stage2_model", "basemodel_on_s2_data"]:
        if model_key_eval not in s2_final_eval_results or model_key_eval not in s2_final_error_grids:
            continue
        metrics_eval = s2_final_eval_results[model_key_eval]
        error_grids_eval = s2_final_error_grids[model_key_eval]
        excel_rows_final_s2.append({'資料來源': f"--- {model_key_eval} (Test Set) ---",
                                '網格座標_R': '', '網格座標_C': '', '經度': '', '緯度': '',
                                'MSE': '', 'MAE': '', 'MAPE': '', 'SMAPE': '', 'FID': ''}) # 添加分隔行

        for flat_idx in range(num_grid_cells_final):
            grid_r_coord, grid_c_coord = 'N/A', 'N/A'
            lon_coord, lat_coord = np.nan, np.nan

            if grid_idx_to_rc_map_s2 and flat_idx in grid_idx_to_rc_map_s2:
                grid_r_coord, grid_c_coord = grid_idx_to_rc_map_s2[flat_idx]

            if sorted_flow_columns_s2 and flat_idx < len(sorted_flow_columns_s2):
                col_name = sorted_flow_columns_s2[flat_idx]
                if sensor_info_lookup_s2 and col_name in sensor_info_lookup_s2:
                    lon_coord = sensor_info_lookup_s2[col_name]['lon']
                    lat_coord = sensor_info_lookup_s2[col_name]['lat']

            row_d = {
                '資料來源': model_key_eval,
                '網格座標_R': grid_r_coord,
                '網格座標_C': grid_c_coord,
                '經度': lon_coord,
                '緯度': lat_coord,
                'MSE': error_grids_eval.get('MSE')[flat_idx] if error_grids_eval.get('MSE') is not None and flat_idx < len(error_grids_eval.get('MSE')) else np.nan,
                'MAE': error_grids_eval.get('MAE')[flat_idx] if error_grids_eval.get('MAE') is not None and flat_idx < len(error_grids_eval.get('MAE')) else np.nan,
                'MAPE': error_grids_eval.get('MAPE')[flat_idx] if error_grids_eval.get('MAPE') is not None and flat_idx < len(error_grids_eval.get('MAPE')) else np.nan,
                'SMAPE': error_grids_eval.get('SMAPE')[flat_idx] if error_grids_eval.get('SMAPE') is not None and flat_idx < len(error_grids_eval.get('SMAPE')) else np.nan,
                'FID': 'N/A' # FID 通常不是針對每個網格計算
            }
            excel_rows_final_s2.append(row_d)

        avg_row_eval = {
            '資料來源': model_key_eval, '網格座標_R': '整體平均', '網格座標_C': '', '經度': '', '緯度': '',
            'MSE': metrics_eval.get('mse', np.nan), 'MAE': metrics_eval.get('mae', np.nan),
            'MAPE': metrics_eval.get('mape', np.nan), 'SMAPE': metrics_eval.get('smape', np.nan),
            'FID': metrics_eval.get('fid', np.nan)
        }
        excel_rows_final_s2.append(avg_row_eval)

    if "stage2_model" in s2_final_eval_results and "basemodel_on_s2_data" in s2_final_eval_results and \
       "stage2_model" in s2_final_error_grids and "basemodel_on_s2_data" in s2_final_error_grids:

        logger.info("計算 Stage2 Model 與 Basemodel 的指標差異...")

        metrics_s2 = s2_final_eval_results["stage2_model"]
        metrics_bm = s2_final_eval_results["basemodel_on_s2_data"]
        error_grids_s2 = s2_final_error_grids["stage2_model"]
        error_grids_bm = s2_final_error_grids["basemodel_on_s2_data"]

        # 添加差異標題行
        excel_rows_final_s2.append({'資料來源': f"--- Difference (Stage2 - Basemodel) ---",
                                '網格座標_R': '', '網格座標_C': '', '經度': '', '緯度': '',
                                'MSE': '', 'MAE': '', 'MAPE': '', 'SMAPE': '', 'FID': ''})

        # 計算並添加每個網格的指標差異
        for flat_idx in range(num_grid_cells_final): # num_grid_cells_final 應該已經在前面定義了
            grid_r_coord, grid_c_coord = 'N/A', 'N/A'
            lon_coord, lat_coord = np.nan, np.nan

            if grid_idx_to_rc_map_s2 and flat_idx in grid_idx_to_rc_map_s2:
                grid_r_coord, grid_c_coord = grid_idx_to_rc_map_s2[flat_idx]

            if sorted_flow_columns_s2 and flat_idx < len(sorted_flow_columns_s2):
                col_name = sorted_flow_columns_s2[flat_idx]
                if sensor_info_lookup_s2 and col_name in sensor_info_lookup_s2:
                    lon_coord = sensor_info_lookup_s2[col_name]['lon']
                    lat_coord = sensor_info_lookup_s2[col_name]['lat']
            
            diff_row_d = {
                '資料來源': "Difference (S2-BM)",
                '網格座標_R': grid_r_coord,
                '網格座標_C': grid_c_coord,
                '經度': lon_coord,
                '緯度': lat_coord,
                'MSE': (error_grids_s2.get('MSE')[flat_idx] - error_grids_bm.get('MSE')[flat_idx])
                        if error_grids_s2.get('MSE') is not None and error_grids_bm.get('MSE') is not None and
                           flat_idx < len(error_grids_s2.get('MSE')) and flat_idx < len(error_grids_bm.get('MSE'))
                        else np.nan,
                'MAE': (error_grids_s2.get('MAE')[flat_idx] - error_grids_bm.get('MAE')[flat_idx])
                        if error_grids_s2.get('MAE') is not None and error_grids_bm.get('MAE') is not None and
                           flat_idx < len(error_grids_s2.get('MAE')) and flat_idx < len(error_grids_bm.get('MAE'))
                        else np.nan,
                'MAPE': (error_grids_s2.get('MAPE')[flat_idx] - error_grids_bm.get('MAPE')[flat_idx])
                        if error_grids_s2.get('MAPE') is not None and error_grids_bm.get('MAPE') is not None and
                           flat_idx < len(error_grids_s2.get('MAPE')) and flat_idx < len(error_grids_bm.get('MAPE'))
                        else np.nan,
                'SMAPE': (error_grids_s2.get('SMAPE')[flat_idx] - error_grids_bm.get('SMAPE')[flat_idx])
                         if error_grids_s2.get('SMAPE') is not None and error_grids_bm.get('SMAPE') is not None and
                            flat_idx < len(error_grids_s2.get('SMAPE')) and flat_idx < len(error_grids_bm.get('SMAPE'))
                         else np.nan,
                'FID': 'N/A' # FID 不是針對每個網格計算
            }
            excel_rows_final_s2.append(diff_row_d)

        # 計算並添加整體平均指標的差異
        diff_avg_row_eval = {
            '資料來源': "Difference (S2-BM)",
            '網格座標_R': '整體平均差異', '網格座標_C': '', '經度': '', '緯度': '',
            'MSE': (metrics_s2.get('mse', np.nan) - metrics_bm.get('mse', np.nan))
                   if not (np.isnan(metrics_s2.get('mse', np.nan)) or np.isnan(metrics_bm.get('mse', np.nan))) else np.nan,
            'MAE': (metrics_s2.get('mae', np.nan) - metrics_bm.get('mae', np.nan))
                   if not (np.isnan(metrics_s2.get('mae', np.nan)) or np.isnan(metrics_bm.get('mae', np.nan))) else np.nan,
            'MAPE': (metrics_s2.get('mape', np.nan) - metrics_bm.get('mape', np.nan))
                    if not (np.isnan(metrics_s2.get('mape', np.nan)) or np.isnan(metrics_bm.get('mape', np.nan))) else np.nan,
            'SMAPE': (metrics_s2.get('smape', np.nan) - metrics_bm.get('smape', np.nan))
                     if not (np.isnan(metrics_s2.get('smape', np.nan)) or np.isnan(metrics_bm.get('smape', np.nan))) else np.nan,
            'FID': (metrics_s2.get('fid', np.nan) - metrics_bm.get('fid', np.nan))
                   if not (np.isnan(metrics_s2.get('fid', np.nan)) or np.isnan(metrics_bm.get('fid', np.nan))) else np.nan, # FID 差異通常也關注
        }
        excel_rows_final_s2.append(diff_avg_row_eval)
        logger.info("指標差異計算並添加到 Excel 數據中。")
    else:
        logger.warning("無法計算指標差異，因為 Stage2 Model 或 Basemodel 的結果缺失。")

    if excel_rows_final_s2:
        df_excel_final_s2 = pd.DataFrame(excel_rows_final_s2)
        excel_column_order_s2 = ['資料來源', '網格座標_R', '網格座標_C', '經度', '緯度', 'MSE', 'MAE', 'MAPE', 'SMAPE', 'FID']
        df_excel_final_s2 = df_excel_final_s2.reindex(columns=excel_column_order_s2)

        excel_final_path_s2 = os.path.join(CONFIG["stage2_model_save_dir"], f"final_test_metrics_detailed_{STAGE2_MODEL_NAME}.xlsx")
        df_excel_final_s2.to_excel(excel_final_path_s2, index=False)
        logger.info(f"Stage2 詳細測試評估指標 (包含差異) 已匯出至: {excel_final_path_s2}")

else:
    logger.warning("Stage2 最終評估的測試數據集為空，跳過評估。")

logger.info(f"===== Stage2 流程全部結束 ({STAGE2_MODEL_NAME}) =====")
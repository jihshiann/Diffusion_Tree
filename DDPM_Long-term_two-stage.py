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
    "basemodel_checkpoint_to_load_for_stage2": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_conditioned_flow_taipei_extra_v2\best_ddpm_model_during_training.pth",

    # === Stage2 特定配置 ===
    "stage2_new_condition_feature_column": "紫外線指數", # 新條件的欄位名
    "stage2_new_conditional_operator": "<=",         # 新條件的運算符
    "stage2_new_conditional_value": 0.0,             # 新條件的閾值
    "stage2_model_name": "Stage2_UVle0_Transfer",    # 第二階段模型的名稱
    "stage2_ddpm_condition_input_channels": 2,       # Stage2 DDPM 的 condition_processor 輸入通道數 (固定為2: bm_out + uv_grid)
    "stage2_checkpoint_path": "best_stage2_model.pth", # Stage2 模型的檢查點檔名 (相對路徑，相對於stage2_model_save_dir)

    # --- DDPM 擴散參數 ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 訓練參數 (Stage2 將優先使用 epochs_stage2, lr_stage2 等，若無則回退到通用版本) ---
    "epochs": 128, 
    "batch_size": 128,
    "lr": 1e-4, # 根據您的需求，Stage2 微調時學習率通常較小
    
    "epochs_stage2": 64, # 可為 Stage2 設定不同的 epoch 數
    "lr_stage2": 5e-5,   # 可為 Stage2 設定不同的學習率

    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "weight_decay": 1e-5,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 4,
    "lr_scheduler_min_lr": 1e-7, 
    "early_stopping_patience": 8,
    "val_calculation_freq": 1, # 建議 Stage2 每個 epoch 都驗證以更好地保存模型

    "resume_from_stage2_checkpoint": False,  # Stage2 訓練是否從自己的檢查點續訓

    # --- 評估參數 ---
    "eval_batch_size": 32,
    "fid_batch_size": 64,
    "fid_num_samples": 128, # 通用FID樣本數
    "fid_num_samples_stage2": 128, # Stage2 FID 計算樣本數 (可與通用相同或不同)


    # --- 路徑與儲存 ---
    # save_dir 將在主腳本中根據 stage2_model_name 重新設定
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
    # raise FileNotFoundError(f"未找到 Basemodel 檢查點檔案: {CONFIG['basemodel_checkpoint_to_load_for_stage2']}") # 如果希望直接中斷

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
        # 這些層的輸出維度與對應特徵圖的通道數匹配
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

        # Decoder path
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
                 # 這個 condition_input_channels 參數將決定 condition_processor 的輸入通道數。
                 # 對於 basemodel (原始)，它是 2 (小時網格+假日網格)。
                 # 對於 stage2_model，它也是 2 (basemodel輸出網格+新條件網格)。
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

        # Condition processor 的輸入通道數由 condition_input_channels 決定
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

    # --- 新增：用於 Basemodel (原始) 的條件準備邏輯 ---
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

        # 這裡假設 condition_processor 的輸入通道數是 2
        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_original_conditional_input_grids: Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels.")

        final_stacked_grids = torch.cat((hour_grids_t, holiday_grids_t), dim=1)
        return final_stacked_grids.to(self.device)

    # --- 用於 Stage2 的條件準備邏輯 (保持不變) ---
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

    # --- 修改 p_losses 以處理兩種條件 ---
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
            # self.logger.debug("p_losses: Using original (scalar hour/holiday) condition mode.")
        elif basemodel_output_grid_batch is not None and new_condition_feature_grid_batch is not None:
            # Stage2 條件模式
            stacked_cond_grids = self._prepare_stage2_condition_grids(
                basemodel_output_grid_batch,
                new_condition_feature_grid_batch
            )
            # self.logger.debug("p_losses: Using stage2 (grid basemodel_out/new_feature) condition mode.")
        else:
            raise ValueError("p_losses: Insufficient or ambiguous condition arguments provided.")

        # 驗證 condition_processor 的輸入通道數
        expected_cond_proc_input_channels = self.condition_processor[0].in_channels
        if stacked_cond_grids.shape[1] != expected_cond_proc_input_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for p_losses. "
                              f"ConditionProcessor expected {expected_cond_proc_input_channels} channels, "
                              f"but got {stacked_cond_grids.shape[1]}.")
        stacked_cond_grids = stacked_cond_grids.to(self.device)
        processed_condition = self.condition_processor(stacked_cond_grids)
        predicted_noise = self.model(x_t_noisy_target, t, processed_condition)
        return F.mse_loss(noise, predicted_noise)

    # --- 修改 sample 以處理兩種條件 ---
    @torch.no_grad()
    def sample(self, batch_size: int,
               # 條件參數 - 擇一提供
               hour_scalars_batch: Optional[torch.Tensor] = None,
               is_holiday_scalars_batch: Optional[torch.Tensor] = None,
               basemodel_output_grid_batch: Optional[torch.Tensor] = None,
               new_condition_feature_grid_batch: Optional[torch.Tensor] = None
               ) -> torch.Tensor:

        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)

        stacked_cond_grids: Optional[torch.Tensor] = None
        # 判斷是哪種條件模式
        if hour_scalars_batch is not None and is_holiday_scalars_batch is not None:
            # Basemodel (原始) 條件模式
            if basemodel_output_grid_batch is not None or new_condition_feature_grid_batch is not None:
                raise ValueError("sample: Cannot provide both scalar (hour/holiday) and grid conditions simultaneously.")
            if hour_scalars_batch.shape[0] != batch_size or is_holiday_scalars_batch.shape[0] != batch_size:
                raise ValueError(f"Original condition batch sizes ({hour_scalars_batch.shape[0]},{is_holiday_scalars_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(
                hour_scalars_batch, is_holiday_scalars_batch
            ).to(self.device)
            # self.logger.debug("sample: Using original (scalar hour/holiday) condition mode.")
        elif basemodel_output_grid_batch is not None and new_condition_feature_grid_batch is not None:
            # Stage2 條件模式
            if basemodel_output_grid_batch.shape[0] != batch_size or new_condition_feature_grid_batch.shape[0] != batch_size:
                raise ValueError(f"Stage2 condition batch sizes ({basemodel_output_grid_batch.shape[0]},{new_condition_feature_grid_batch.shape[0]}) != requested batch_size ({batch_size})")
            stacked_cond_grids = self._prepare_stage2_condition_grids(
                basemodel_output_grid_batch,
                new_condition_feature_grid_batch
            )
            # self.logger.debug("sample: Using stage2 (grid basemodel_out/new_feature) condition mode.")
        else:
            raise ValueError("sample: Insufficient or ambiguous condition arguments provided.")

        # 驗證 condition_processor 的輸入通道數
        expected_cond_proc_input_channels = self.condition_processor[0].in_channels
        if stacked_cond_grids.shape[1] != expected_cond_proc_input_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for sampling. "
                              f"ConditionProcessor expected {expected_cond_proc_input_channels} channels, "
                              f"but got {stacked_cond_grids.shape[1]}.")

        processed_conditions = self.condition_processor(stacked_cond_grids)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="DDPM Unified Sampling", total=self.timesteps, leave=False):
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


def apply_condition_to_dataframe(df: pd.DataFrame,
                                 condition_feature: str,
                                 condition_operator: str,
                                 condition_value: Any,
                                 logger_instance: logging.Logger) -> Optional[pd.DataFrame]:
    """
    對 DataFrame 應用條件進行篩選。
    """
    logger_instance.info(f"開始篩選 DataFrame。原始大小: {df.shape}")
    filtered_df = df.copy()

    if condition_feature not in filtered_df.columns:
        logger_instance.error(f"錯誤：條件特徵 '{condition_feature}' 不在 DataFrame 的欄位中。")
        return None

    original_dtype = filtered_df[condition_feature].dtype
    numeric_column = pd.to_numeric(filtered_df[condition_feature], errors='coerce')

    if numeric_column.isnull().all() and not pd.api.types.is_numeric_dtype(original_dtype) and df[condition_feature].notnull().any():
        logger_instance.warning(f"警告：條件特徵 '{condition_feature}' (原類型: {original_dtype}) 在嘗試轉換為數值後所有值均為 NaN。請檢查此特徵是否適合數值比較。")
    elif numeric_column.isnull().sum() > 0:
         logger_instance.warning(f"警告：條件特徵 '{condition_feature}' 包含 {numeric_column.isnull().sum()} 個無法轉換為數值的項目。這些項目在比較中將被視為 NaN。")

    try:
        val_for_comp = float(condition_value)
        is_numeric_comp = True
    except ValueError:
        val_for_comp = condition_value
        is_numeric_comp = False

    if condition_operator == "<=":
        if not is_numeric_comp:
            logger_instance.error(f"錯誤：運算符 '{condition_operator}' 需要數值比較，但條件值 '{condition_value}' 不是數值。")
            return None
        mask = numeric_column <= val_for_comp
    elif condition_operator == ">=":
        if not is_numeric_comp:
            logger_instance.error(f"錯誤：運算符 '{condition_operator}' 需要數值比較，但條件值 '{condition_value}' 不是數值。")
            return None
        mask = numeric_column >= val_for_comp
    elif condition_operator == "<":
        if not is_numeric_comp:
            logger_instance.error(f"錯誤：運算符 '{condition_operator}' 需要數值比較，但條件值 '{condition_value}' 不是數值。")
            return None
        mask = numeric_column < val_for_comp
    elif condition_operator == ">":
        if not is_numeric_comp:
            logger_instance.error(f"錯誤：運算符 '{condition_operator}' 需要數值比較，但條件值 '{condition_value}' 不是數值。")
            return None
        mask = numeric_column > val_for_comp
    elif condition_operator == "==":
        if is_numeric_comp and not isinstance(val_for_comp, str): # 優先數值比較 (除非條件值本身是字串)
            mask = numeric_column == val_for_comp
        else:
            mask = filtered_df[condition_feature].astype(str) == str(val_for_comp)
    elif condition_operator == "!=":
        if is_numeric_comp and not isinstance(val_for_comp, str):
            mask = numeric_column != val_for_comp
        else:
            mask = filtered_df[condition_feature].astype(str) != str(val_for_comp)
    else:
        logger_instance.error(f"錯誤：不支援的運算符: {condition_operator}")
        return None

    filtered_df = filtered_df[mask.fillna(False)]

    if filtered_df.empty:
        logger_instance.warning(f"警告：應用條件 '{condition_feature} {condition_operator} {condition_value}' 後，沒有數據滿足。")
    else:
        logger_instance.info(f"篩選完成。篩選後 DataFrame 大小: {filtered_df.shape}")
    return filtered_df


def create_stage2_model_from_basemodel_checkpoint(
                               basemodel_checkpoint_path: str,
                               config_for_stage2_model: Dict[str, Any],
                               device: str
                               ) -> DDPM3D:
    logger.info(f"從 Basemodel 檢查點 {basemodel_checkpoint_path} 創建並初始化 Stage2 模型...")
    
    chkpt_basemodel = torch.load(basemodel_checkpoint_path, map_location=device, weights_only = False)
    if 'ddpm_state_dict' not in chkpt_basemodel:
        raise KeyError(f"Basemodel 檢查點 {basemodel_checkpoint_path} 中未找到 'ddpm_state_dict'。")
    
    basemodel_original_config = chkpt_basemodel.get('config', config_for_stage2_model)

    # Stage2 模型的 UNet 架構應與 basemodel 一致
    stage2_unet = UNet3D(
        input_image_channels=basemodel_original_config.get("image_channels", config_for_stage2_model["image_channels"]),
        base_channels=basemodel_original_config.get("base_channels_unet", config_for_stage2_model["base_channels_unet"]),
        time_emb_dim=basemodel_original_config.get("time_emb_dim", config_for_stage2_model["time_emb_dim"]),
        condition_encode_dim=basemodel_original_config.get("condition_encode_dim", config_for_stage2_model["condition_encode_dim"]),
        dropout_rate=basemodel_original_config.get("unet_dropout_rate", config_for_stage2_model.get("unet_dropout_rate", 0.05))
    ).to(device)

    # Stage2 DDPM 的 condition_input_channels 固定為2 (或由 stage2 特定配置決定)
    # 這個值來自 config_for_stage2_model，它應該已經被設定為期望的值 (例如2)
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
    # 這裡假設 basemodel 的 condition_processor 也是2通道輸入 (config_basemodel_original["condition_input_channels"] == 2)
    # 因此完整的 state_dict 可以直接載入l
    try:
        stage2_model_instance.load_state_dict(chkpt_basemodel['ddpm_state_dict'])
        logger.info("Stage2 模型權重從 Basemodel 完整遷移完成。")
    except RuntimeError as e:
        logger.error(f"直接載入 Basemodel state_dict 到 Stage2 模型失敗: {e}")
        logger.warning("嘗試僅載入 UNet (model) 部分的權重...")
        stage2_model_instance.model.load_state_dict(chkpt_basemodel['ddpm_state_dict']['model']) # 假設鍵是 'model'
        logger.info("僅 UNet 權重從 Basemodel 遷移完成。Condition Processor 將使用隨機初始化權重。")
        # 這種情況下，需要確保 CONFIG 中 stage2_ddpm_condition_input_channels 與 basemodel 的不同，
        # 或者 condition_processor 的架構不同。

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
                 df_for_stage2_processing: pd.DataFrame, # 改名以反映其未被紫外線篩選
                 basemodel_outputs_for_samples_np: np.ndarray,
                 config: Dict[str, Any],
                 original_sorted_flow_columns: List[str],
                 mode: str = 'train',
                 stage2_target_stats_from_train: Optional[Dict[str, float]] = None,
                 stage2_avg_flow_map_dict_from_train: Optional[Dict[Tuple, np.ndarray]] = None,
                 new_cond_feature_norm_stats_from_train: Optional[Dict[str, float]] = None
                 ):
        super().__init__()
        self.df_s2 = df_for_stage2_processing.reset_index(drop=True) # df_s2 現在是更廣泛的數據
        self.basemodel_outputs_np = basemodel_outputs_for_samples_np
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        self.H = config["H"]
        self.W = config["W"]
        self.D = config.get("D", 1)
        self.image_channels_target = config.get("image_channels", 1)

        self.uv_col_name = config["stage2_new_condition_feature_column"] # 沿用，但理解為要分類的列
        self.uv_op = config["stage2_new_conditional_operator"]         # 運算符
        self.uv_val = config["stage2_new_conditional_value"]           # 閾值
        self.sorted_flow_columns = original_sorted_flow_columns

        # ... (維度檢查等保持不變) ...
        if self.uv_col_name not in self.df_s2.columns:
             raise ValueError(f"Stage2Dataset: 新條件特徵的原始欄位 '{self.uv_col_name}' 不在 DataFrame 中。")


        dt_series = pd.to_datetime(self.df_s2['時間'])
        self.hours_for_target_np = dt_series.dt.hour.values
        # ... (假日處理保持不變) ...
        if self.df_s2['holiday'].dtype == bool:
            self.is_holiday_for_target_np = self.df_s2['holiday'].astype(int).values
        elif pd.api.types.is_numeric_dtype(self.df_s2['holiday']):
             self.is_holiday_for_target_np = self.df_s2['holiday'].fillna(0).astype(bool).astype(int).values
        else: 
            holiday_map = {'是': 1, 'true': 1, '1': 1, 'yes': 1, 'y': 1, '否': 0, 'false': 0, '0': 0, 'no': 0, 'n': 0}
            self.is_holiday_for_target_np = self.df_s2['holiday'].astype(str).str.lower().map(holiday_map).fillna(0).astype(int).values
        
        # 獲取紫外線指數的原始數值 (用於正規化並作為 Stage2 模型的條件2輸入)
        self.uv_original_values_np = pd.to_numeric(self.df_s2[self.uv_col_name], errors='coerce').values

        # --- 根據紫外線指數創建分類特徵 (uv_category_for_target_np) ---
        # 這個分類將用於 _calculate_stage2_target_flows 的 groupby
        numeric_uv_vals_for_category = pd.to_numeric(self.df_s2[self.uv_col_name], errors='coerce')
        # 簡單示例：分為兩類 (<= threshold vs > threshold)
        # 您可以根據需求定義更複雜的分類邏輯，例如多分箱
        if self.uv_op == "<=":
            # 類別 0: 紫外線 <= 閾值
            # 類別 1: 紫外線 > 閾值 (或 NaN) -> 我們將 NaN 也歸為一類或特定處理
            # 為了簡化，這裡假設 NaN 的情況比較少，或者在 groupby 時 nanmean 會處理
            self.uv_category_for_target_np = (numeric_uv_vals_for_category <= float(self.uv_val)).astype(int)
            self.logger.info(f"Stage2Dataset: 紫外線分類邏輯 -> '{self.uv_col_name}' <= {self.uv_val} 為類別 1，否則為類別 0 (反轉一下，小的為0，大的為1)")
            self.uv_category_for_target_np = (~(numeric_uv_vals_for_category <= float(self.uv_val))).astype(int) # 小於等於為0，大於為1

        elif self.uv_op == ">":
            # 類別 0: 紫外線 > 閾值
            # 類別 1: 紫外線 <= 閾值 (或 NaN)
            self.uv_category_for_target_np = (numeric_uv_vals_for_category > float(self.uv_val)).astype(int)
        else:
            self.logger.warning(f"Stage2Dataset: 未明確處理紫外線運算符 '{self.uv_op}'，默認分類為 ({self.uv_col_name} <= {self.uv_val}) 為類別0，否則為類別1。")
            self.uv_category_for_target_np = (~(numeric_uv_vals_for_category <= float(self.uv_val))).astype(int)
        
        # 打印一些分類統計
        unique_cats, counts_cats = np.unique(self.uv_category_for_target_np, return_counts=True)
        self.logger.info(f"Stage2Dataset: 生成的紫外線分類 (uv_category_for_target_np) 分佈: {dict(zip(unique_cats, counts_cats))}")


        # --- 紫外線指數的正規化 (用於 Stage2 模型的條件2輸入) ---
        if self.mode == 'train':
            valid_uv_values = self.uv_original_values_np[~np.isnan(self.uv_original_values_np)]
            if len(valid_uv_values) > 0:
                self.uv_feature_mean = np.mean(valid_uv_values)
                self.uv_feature_std = np.std(valid_uv_values)
            else:
                self.uv_feature_mean = 0.0
                self.uv_feature_std = 1.0
            if self.uv_feature_std < 1e-6: self.uv_feature_std = 1.0
            self.norm_stats_new_cond_feature = {'mean': self.uv_feature_mean, 'std': self.uv_feature_std}
            self.logger.info(f"Stage2 訓練集紫外線特徵 '{self.uv_col_name}' 正規化統計: Mean={self.uv_feature_mean:.4f}, Std={self.uv_feature_std:.4f}")
        else:
            # ... (從 new_cond_feature_norm_stats_from_train 加載 uv_feature_mean, uv_feature_std 的邏輯不變) ...
            if new_cond_feature_norm_stats_from_train is None:
                raise ValueError(f"Stage2 val/test mode 需要從訓練集傳入 new_cond_feature_norm_stats。")
            self.norm_stats_new_cond_feature = new_cond_feature_norm_stats_from_train
            self.uv_feature_mean = self.norm_stats_new_cond_feature['mean']
            self.uv_feature_std = self.norm_stats_new_cond_feature['std']
            if self.uv_feature_std < 1e-6: self.uv_feature_std = 1.0

        # --- Stage2 目標流量的正規化統計量計算 (基於新的 groupby) ---
        if self.mode == 'train':
            self.average_flow_map_dict_s2 = self._calculate_stage2_target_flows() # 現在會基於 (hr, hol, uv_cat)
            # ... (後續的 target_mean_s2, target_std_s2 計算邏輯不變，但數據源變了) ...
            all_avg_flows_list = [flow for flow in self.average_flow_map_dict_s2.values() if flow is not None]
            if not all_avg_flows_list:
                 self.logger.warning("Stage2 訓練集: 未計算出任何目標平均流量。目標流量正規化統計量將為0和1。")
                 self.target_mean_s2 = 0.0
                 self.target_std_s2 = 1.0
            else:
                all_avg_flows_np = np.stack(all_avg_flows_list)
                self.target_mean_s2 = np.mean(all_avg_flows_np)
                self.target_std_s2 = np.std(all_avg_flows_np)
            if self.target_std_s2 < 1e-5: self.target_std_s2 = 1e-5
            self.norm_stats_target_s2 = {'mean': self.target_mean_s2, 'std': self.target_std_s2}
            self.logger.info(f"Stage2 訓練集目標流量正規化統計 (基於小時,假日,紫外線分類): Mean={self.target_mean_s2:.4f}, Std={self.target_std_s2:.4f}")
        else:
            # ... (從 stage2_avg_flow_map_dict_from_train, stage2_target_stats_from_train 加載的邏輯不變) ...
            if stage2_avg_flow_map_dict_from_train is None or stage2_target_stats_from_train is None:
                raise ValueError("Stage2 val/test mode 需要從訓練集傳入 stage2_avg_flow_map_dict 和 stage2_target_stats。")
            self.average_flow_map_dict_s2 = stage2_avg_flow_map_dict_from_train
            self.norm_stats_target_s2 = stage2_target_stats_from_train
            self.target_mean_s2 = self.norm_stats_target_s2['mean']
            self.target_std_s2 = self.norm_stats_target_s2['std']
            if self.target_std_s2 < 1e-5: self.target_std_s2 = 1e-5


    def _calculate_stage2_target_flows(self) -> Dict[Tuple[int, int, int], np.ndarray]: #鍵現在是 (hr, hol, uv_cat)
        self.logger.info("Stage2: 計算複合條件 (小時, 假日, 紫外線分類) 的目標平均流量...")
        avg_flows = {}
        flow_data_for_calc = self.df_s2[self.sorted_flow_columns].values.astype(np.float32)

        grouping_df = pd.DataFrame({
            'hour': self.hours_for_target_np,
            'is_holiday': self.is_holiday_for_target_np,
            'uv_category_for_target': self.uv_category_for_target_np # 使用新的紫外線分類
        })
        # 根據小時, 假日, 和紫外線分類進行分組
        grouped = grouping_df.groupby(['hour', 'is_holiday', 'uv_category_for_target'])

        if not grouped.groups:
            self.logger.warning("Stage2Dataset: 無法根據 (小時, 假日, 紫外線分類) 對資料進行分組。")
            return {}

        self.logger.info("Stage2 Target Calculation: 樣本數分佈如下 (hour, is_holiday, uv_category): count")
        for (hr, is_hol, uv_cat), group_indices in grouped.groups.items():
            count = len(group_indices)
            self.logger.info(f"  - ({hr:02d}, {is_hol}, {uv_cat}): {count} samples")

        for (hr, is_hol, uv_cat), group_indices in grouped.groups.items():
            if len(group_indices) == 0: continue
            group_flows_flat = flow_data_for_calc[group_indices]
            mean_flow_flat = np.nanmean(group_flows_flat, axis=0)
            mean_flow_flat[np.isnan(mean_flow_flat)] = 0
            avg_flows[(hr, int(is_hol), int(uv_cat))] = mean_flow_flat.reshape(self.H, self.W) # 鍵使用紫外線分類
        self.logger.info(f"Stage2: 計算完成 {len(avg_flows)} 個 (小時,假日,紫外線分類) 條件的目標平均流量圖。")
        return avg_flows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bm_output_grid_np_sample = self.basemodel_outputs_np[idx]
        condition_grid_1_tensor = torch.from_numpy(bm_output_grid_np_sample.astype(np.float32)) # 假設已正規化

        # 獲取並正規化紫外線指數作為條件2
        original_uv_value = self.uv_original_values_np[idx]
        if np.isnan(original_uv_value):
            normalized_uv_value = 0.0 # 或其他填充策略的正規化結果
        else:
            normalized_uv_value = (original_uv_value - self.uv_feature_mean) / self.uv_feature_std
        
        condition_grid_2_tensor = torch.full(
            (1, self.D, self.H, self.W),
            float(normalized_uv_value),
            dtype=torch.float32
        )

        # 獲取目標流量
        hr = self.hours_for_target_np[idx]
        is_hol = self.is_holiday_for_target_np[idx]
        uv_cat = self.uv_category_for_target_np[idx] # 使用生成的紫外線分類
        target_key = (hr, is_hol, uv_cat)
        
        target_avg_flow_s2_np = self.average_flow_map_dict_s2.get(target_key)
        if target_avg_flow_s2_np is None:
            self.logger.debug(f"Stage2Dataset (idx {idx}): 未找到目標鍵 {target_key}，使用零值網格。")
            target_avg_flow_s2_np = np.zeros((self.H, self.W), dtype=np.float32)
        
        std_val_safe_target = self.target_std_s2 if self.target_std_s2 > 1e-6 else 1.0
        norm_target_s2_np = (target_avg_flow_s2_np - self.target_mean_s2) / std_val_safe_target
        
        target_flow_tensor = torch.from_numpy(norm_target_s2_np).float().reshape(
            self.image_channels_target, self.D, self.H, self.W
        )
        
        # 返回原始的小時和假日，供 Basemodel 在評估時使用其原始條件
        original_hour_scalar_tensor = torch.tensor(hr, dtype=torch.long)
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
    save_dir = config["save_dir"]
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
             # 可以選擇用特定值（如0或nan）替換，或跳過繪圖
             # 這裡用nan，讓colorbar處理
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
    stage2_model_trained: DDPM3D,
    basemodel_eval_instance: DDPM3D,
    dataloader_s2: DataLoader,
    inception_model_fid: nn.Module,
    config: Dict[str, Any],
    max_samples_for_fid: Optional[int] = None,
    prefix: str = "stage2_eval"
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    logger.info(f"===== 開始 Stage2 模型評估 (比較 {prefix}) =====")
    stage2_model_trained.eval()
    basemodel_eval_instance.eval()
    inception_model_fid.eval()

    dataset_s2_obj = dataloader_s2.dataset
    if not hasattr(dataset_s2_obj, 'norm_stats_target_s2'):
        raise AttributeError("Stage2 Dataloader 的 dataset缺少 'norm_stats_target_s2'。")
    s2_target_mean = dataset_s2_obj.norm_stats_target_s2['mean']
    s2_target_std = dataset_s2_obj.norm_stats_target_s2['std']
    if s2_target_std < 1e-6: s2_target_std = 1.0

    basemodel_orig_mean = config["cached_basemodel_mean"]
    basemodel_orig_std = config["cached_basemodel_std"]
    if basemodel_orig_std < 1e-6: basemodel_orig_std = 1.0

    all_s2_generated_denorm_list: List[torch.Tensor] = []
    all_bm_generated_denorm_on_s2_data_list: List[torch.Tensor] = []
    all_s2_target_denorm_list: List[torch.Tensor] = []
    
    all_s2_generated_norm_for_fid_list: List[torch.Tensor] = []
    all_bm_generated_norm_for_fid_on_s2_data_list: List[torch.Tensor] = []
    all_s2_target_norm_for_fid_list: List[torch.Tensor] = []

    max_fid_samples_actual = len(dataset_s2_obj)
    if max_samples_for_fid is not None:
        max_fid_samples_actual = min(max_samples_for_fid, max_fid_samples_actual)

    pbar_s2_eval = tqdm(dataloader_s2, desc=f"Stage2 評估 ({prefix})", leave=False)
    for target_s2_norm, bm_out_grid_cond, uv_grid_cond, orig_hr_s, orig_is_hol_s in pbar_s2_eval:
        current_batch_size = target_s2_norm.shape[0]
        target_s2_norm = target_s2_norm.to(config["device"])
        bm_out_grid_cond = bm_out_grid_cond.to(config["device"])
        uv_grid_cond = uv_grid_cond.to(config["device"])
        orig_hr_s = orig_hr_s.to(config["device"])
        orig_is_hol_s = orig_is_hol_s.to(config["device"])

        # 1. Stage2 模型生成 (使用網格條件)
        s2_generated_norm = stage2_model_trained.sample(
            batch_size=current_batch_size,
            basemodel_output_grid_batch=bm_out_grid_cond, # 條件1
            new_condition_feature_grid_batch=uv_grid_cond   # 條件2
        )
        s2_generated_denorm = s2_generated_norm * s2_target_std + s2_target_mean
        all_s2_generated_denorm_list.append(s2_generated_denorm.cpu())

        # 2. Basemodel 在相同原始條件(小時,假日)下的生成
        #    basemodel_eval_instance 的 sample 方法期望的是小時/假日純量，
        #    其內部 _prepare_conditional_input_grids 會將它們轉換成2通道網格。
        bm_generated_norm_on_s2_conditions = basemodel_eval_instance.sample(
            batch_size=current_batch_size,
            hour_scalars_batch=orig_hr_s,
            is_holiday_scalars_batch=orig_is_hol_s
        )
        bm_generated_denorm_on_s2_conditions = bm_generated_norm_on_s2_conditions * basemodel_orig_std + basemodel_orig_mean
        all_bm_generated_denorm_on_s2_data_list.append(bm_generated_denorm_on_s2_conditions.cpu())
        
        s2_target_denorm = target_s2_norm * s2_target_std + s2_target_mean
        all_s2_target_denorm_list.append(s2_target_denorm.cpu())

        samples_collected_so_far = sum(s.shape[0] for s in all_s2_generated_norm_for_fid_list)
        if samples_collected_so_far < max_fid_samples_actual:
            remaining_needed_fid = max_fid_samples_actual - samples_collected_so_far
            samples_to_add_fid = min(current_batch_size, remaining_needed_fid)
            if samples_to_add_fid > 0:
                all_s2_generated_norm_for_fid_list.append(s2_generated_norm[:samples_to_add_fid].cpu())
                all_s2_target_norm_for_fid_list.append(target_s2_norm[:samples_to_add_fid].cpu())
                all_bm_generated_norm_for_fid_on_s2_data_list.append(bm_generated_norm_on_s2_conditions[:samples_to_add_fid].cpu())

    if not all_s2_target_denorm_list:
        logger.warning(f"Stage2 評估 ({prefix}): 無數據處理。")
        nan_metrics = {"mse": float('nan'), "mae": float('nan'), "mape": float('nan'), "smape": float('nan'), "fid": float('nan')}
        nan_grids = {m: np.full((config["H"] * config["W"],), np.nan) for m in ['MSE','MAE','MAPE','SMAPE']}
        return {"stage2_model": nan_metrics, "basemodel_on_s2_data": nan_metrics}, \
               {"stage2_model": nan_grids, "basemodel_on_s2_data": nan_grids}

    s2_target_all_t = torch.cat(all_s2_target_denorm_list, dim=0)
    s2_generated_all_t = torch.cat(all_s2_generated_denorm_list, dim=0)
    bm_generated_all_on_s2_data_t = torch.cat(all_bm_generated_denorm_on_s2_data_list, dim=0)
    logger.info(f"s2_target_all_t (反正規化後的Stage2目標) shape: {s2_target_all_t.shape}")
    logger.info(f"s2_target_all_t min: {torch.min(s2_target_all_t).item()}, max: {torch.max(s2_target_all_t).item()}, mean: {torch.mean(s2_target_all_t).item()}")
    s2_target_abs = torch.abs(s2_target_all_t)
    logger.info(f"s2_target_all_t num_zeros (==0): {torch.sum(s2_target_all_t == 0).item()}")
    logger.info(f"s2_target_all_t num_near_zeros (<0.001): {torch.sum(s2_target_abs < 0.001).item()}")
    logger.info(f"s2_target_all_t num_small (<1): {torch.sum(s2_target_abs < 1.0).item()}")
    logger.info(f"s2_target_all_t num_elements: {s2_target_all_t.numel()}")
    logger.info(f"s2_generated_all_t (反正規化後的Stage2預測) shape: {s2_generated_all_t.shape}")
    logger.info(f"s2_generated_all_t min: {torch.min(s2_generated_all_t).item()}, max: {torch.max(s2_generated_all_t).item()}, mean: {torch.mean(s2_generated_all_t).item()}")
    
    epsilon = 1e-6
    results = {}
    error_grids_all_models = {}

    for model_name, pred_t in [("stage2_model", s2_generated_all_t), 
                               ("basemodel_on_s2_data", bm_generated_all_on_s2_data_t)]:
        mse = F.mse_loss(pred_t, s2_target_all_t).item()
        mae = F.l1_loss(pred_t, s2_target_all_t).item()
        mape_tensor = torch.abs((s2_target_all_t - pred_t) / (torch.abs(s2_target_all_t) + epsilon)) * 100
        mape = torch.mean(mape_tensor[torch.isfinite(mape_tensor)]).item() if torch.isfinite(mape_tensor).any() else float('inf')
        
        smape_num = torch.abs(pred_t - s2_target_all_t)
        smape_den = (torch.abs(s2_target_all_t) + torch.abs(pred_t)) / 2.0 + epsilon # 修正分母
        smape_tensor = (smape_num / smape_den) * 100 
        smape = torch.mean(smape_tensor[torch.isfinite(smape_tensor)]).item() if torch.isfinite(smape_tensor).any() else float('inf')

        fid = float('nan')
        current_generated_norm_for_fid_list_eval: List[torch.Tensor] = []
        if model_name == "stage2_model":
            current_generated_norm_for_fid_list_eval = all_s2_generated_norm_for_fid_list
        elif model_name == "basemodel_on_s2_data":
             current_generated_norm_for_fid_list_eval = all_bm_generated_norm_for_fid_on_s2_data_list

        if current_generated_norm_for_fid_list_eval and all_s2_target_norm_for_fid_list:
            gen_fid_tensor = torch.cat(current_generated_norm_for_fid_list_eval, dim=0)[:max_fid_samples_actual]
            real_fid_tensor = torch.cat(all_s2_target_norm_for_fid_list, dim=0)[:max_fid_samples_actual]
            num_fid = min(gen_fid_tensor.shape[0], real_fid_tensor.shape[0])
            if num_fid > 1:
                logger.info(f"Calculating FID for {model_name} (vs S2 target) on {num_fid} samples...")
                act_gen = get_activations(gen_fid_tensor, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                act_real = get_activations(real_fid_tensor, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                if act_gen.shape[0] > 1 and act_real.shape[0] > 1:
                    fid = calculate_fid(act_real, act_gen)
            else: logger.warning(f"FID for {model_name}: Insufficient samples ({num_fid}).")
        else: logger.warning(f"FID for {model_name}: FID sample lists empty.")
        
        results[model_name] = {"mse": mse, "mae": mae, "mape": mape, "smape": smape, "fid": fid if np.isfinite(fid) else float('nan')}
        logger.info(f"Metrics for {model_name} ({prefix}): {results[model_name]}")

        if pred_t.ndim == 5 and pred_t.shape[-3:] == (config.get("D",1), config["H"], config["W"]): # NCDHW
            pred_squeezed_for_grid_error = pred_t.squeeze(1).squeeze(1) # -> N, H, W
            target_squeezed_for_grid_error = s2_target_all_t.squeeze(1).squeeze(1) # -> N, H, W

            mse_g = torch.mean((pred_squeezed_for_grid_error - target_squeezed_for_grid_error)**2, dim=0).cpu().numpy().flatten()
            mae_g = torch.mean(torch.abs(pred_squeezed_for_grid_error - target_squeezed_for_grid_error), dim=0).cpu().numpy().flatten()
            
            mape_g_t_flat = torch.abs((target_squeezed_for_grid_error - pred_squeezed_for_grid_error) / (torch.abs(target_squeezed_for_grid_error) + epsilon)) * 100
            mape_g = torch.mean(mape_g_t_flat, dim=0).cpu().numpy().flatten()
            
            smape_n_g_flat = torch.abs(pred_squeezed_for_grid_error - target_squeezed_for_grid_error)
            smape_d_g_flat = (torch.abs(target_squeezed_for_grid_error) + torch.abs(pred_squeezed_for_grid_error))/2.0 + epsilon
            smape_g_t_flat = (smape_n_g_flat / smape_d_g_flat) * 100
            smape_g = torch.mean(smape_g_t_flat, dim=0).cpu().numpy().flatten()

            if len(mse_g) == config["H"] * config["W"]:
                error_grids_all_models[model_name] = {'MSE': mse_g, 'MAE': mae_g, 'MAPE': mape_g, 'SMAPE': smape_g}
        else:
            logger.warning(f"Prediction tensor shape mismatch for per-grid metrics ({model_name}). Pred shape: {pred_t.shape}")

    logger.info(f"Generating visualizations for Stage2 evaluation ({prefix})...")
    num_samples_to_plot_viz = min(1, s2_target_all_t.shape[0]) 
    for i in range(num_samples_to_plot_viz):
        visualize_stage2_comparison(
            stage2_model_pred_denorm=s2_generated_all_t[i:i+1].clone().cpu(),
            basemodel_pred_denorm=bm_generated_all_on_s2_data_t[i:i+1].clone().cpu(),
            target_denorm=s2_target_all_t[i:i+1].clone().cpu(),
            config=config, sample_idx=i, prefix=f"{prefix}_sample{i}"
        )
    if s2_target_all_t.shape[0] > 0: # Plot average only if there's data
        visualize_stage2_comparison(
            stage2_model_pred_denorm=torch.mean(s2_generated_all_t, dim=0, keepdim=True).clone().cpu(),
            basemodel_pred_denorm=torch.mean(bm_generated_all_on_s2_data_t, dim=0, keepdim=True).clone().cpu(),
            target_denorm=torch.mean(s2_target_all_t, dim=0, keepdim=True).clone().cpu(),
            config=config, sample_idx=None, prefix=f"{prefix}_avg"
        )

    if "stage2_model" in error_grids_all_models and error_grids_all_models["stage2_model"]: # 確保有誤差數據
        plot_grid_with_error_long_term(
            dataset_s2_obj, # 這個參數雖然傳遞了，但函數內部未使用其網格屬性
            error_grids_all_models["stage2_model"],
            config, # config 參數包含了所需的網格映射資訊
            f"{prefix}_stage2"
        )
    if "basemodel_on_s2_data" in error_grids_all_models and error_grids_all_models["basemodel_on_s2_data"]: # 確保有誤差數據
        plot_grid_with_error_long_term(
            dataset_s2_obj, # 同上
            error_grids_all_models["basemodel_on_s2_data"],
            config, # config 參數包含了所需的網格映射資訊
            f"{prefix}_basemodel"
        )

    if "stage2_model" in error_grids_all_models and "basemodel_on_s2_data" in error_grids_all_models:
        s2_errors = error_grids_all_models["stage2_model"]      # Dict[str, np.ndarray]
        bm_errors = error_grids_all_models["basemodel_on_s2_data"] # Dict[str, np.ndarray]
        
        error_metrics_difference_grids = {} # 用於存儲差異指標網格

        for metric_key in ['MSE', 'MAE', 'MAPE', 'SMAPE']:
            if metric_key in s2_errors and metric_key in bm_errors:
                s2_metric_grid = s2_errors[metric_key]
                bm_metric_grid = bm_errors[metric_key]

                if isinstance(s2_metric_grid, np.ndarray) and isinstance(bm_metric_grid, np.ndarray) and \
                   s2_metric_grid.shape == bm_metric_grid.shape and \
                   s2_metric_grid.shape[0] == config["H"] * config["W"]:
                    
                    # 計算差異： Stage2 Error - Basemodel Error
                    # 正值表示 Stage2 在該網格的該指標上誤差更大 (表現更差)
                    # 負值表示 Stage2 在該網格的該指標上誤差更小 (表現更好)
                    difference_grid = s2_metric_grid - bm_metric_grid
                    error_metrics_difference_grids[f"Diff_{metric_key}_(S2-BM)"] = difference_grid
                else:
                    logger.warning(f"無法計算指標 '{metric_key}' 的差異網格，"
                                   f"因為 Stage2 或 Basemodel 的誤差網格缺失、類型錯誤或形狀不匹配。")
            else:
                logger.warning(f"指標 '{metric_key}' 在 Stage2 或 Basemodel 的誤差網格中缺失，無法計算差異。")

        if error_metrics_difference_grids:
            logger.info(f"為 Stage2 vs Basemodel 的誤差指標差異生成地理熱力圖 ({prefix})...")
            plot_grid_with_error_long_term(
                dataset_s2_obj,  # 或 config，取決於 plot_grid_with_error_long_term 的實現
                error_metrics_difference_grids,
                config,
                f"{prefix}_diff_S2_minus_BM" # 新的檔名前綴
            )
        else:
            logger.info(f"沒有可繪製的 Stage2 vs Basemodel 誤差指標差異網格 ({prefix})。")

    return results, error_grids_all_models # 保持函數原始返回

def visualize_stage2_comparison(
    stage2_model_pred_denorm: torch.Tensor,
    basemodel_pred_denorm: torch.Tensor,
    target_denorm: torch.Tensor,
    config: Dict[str, Any],
    sample_idx: Optional[int],
    prefix: str
):
    save_dir = config["stage2_model_save_dir"] # 儲存到 Stage2 特定目錄
    os.makedirs(save_dir, exist_ok=True)
    title_suffix = f"sample_{sample_idx}" if sample_idx is not None else "avg_all_samples"
    
    s2_pred = stage2_model_pred_denorm.squeeze().cpu().numpy() # Squeeze C,D if they are 1
    bm_pred = basemodel_pred_denorm.squeeze().cpu().numpy()
    target = target_denorm.squeeze().cpu().numpy()

    diff_s2_target = s2_pred - target
    diff_bm_target = bm_pred - target
    diff_s2_bm = s2_pred - bm_pred

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Stage2 Model vs Basemodel Comparison ({title_suffix}) - {prefix}", fontsize=16)

    # 確保vmin和vmax至少有一個小的差異，避免imshow報錯
    vmax_list = [np.max(target), np.max(s2_pred), np.max(bm_pred)]
    vmin_list = [np.min(target), np.min(s2_pred), np.min(bm_pred)]
    common_vmax = max(vmax_list) if vmax_list else 1.0
    common_vmin = min(vmin_list) if vmin_list else 0.0
    if common_vmax <= common_vmin: common_vmax = common_vmin + 1e-5


    im = axes[0, 0].imshow(target, cmap='viridis', vmin=common_vmin, vmax=common_vmax)
    axes[0, 0].set_title(f"Target (Composite Cond.)")
    axes[0, 0].axis('off'); fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(bm_pred, cmap='viridis', vmin=common_vmin, vmax=common_vmax)
    axes[0, 1].set_title(f"Basemodel Output (Orig. Cond.)")
    axes[0, 1].axis('off'); fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(s2_pred, cmap='viridis', vmin=common_vmin, vmax=common_vmax)
    axes[0, 2].set_title(f"Stage2 Model Output")
    axes[0, 2].axis('off'); fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    diff_cmap = 'coolwarm'
    max_abs_diff_val = max(np.max(np.abs(diff_s2_target)), np.max(np.abs(diff_bm_target)), np.max(np.abs(diff_s2_bm)), 1e-5)

    im = axes[1, 0].imshow(diff_s2_target, cmap=diff_cmap, vmin=-max_abs_diff_val, vmax=max_abs_diff_val)
    axes[1, 0].set_title(f"Diff: Stage2 - Target (MAE: {np.mean(np.abs(diff_s2_target)):.2f})")
    axes[1, 0].axis('off'); fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im = axes[1, 1].imshow(diff_bm_target, cmap=diff_cmap, vmin=-max_abs_diff_val, vmax=max_abs_diff_val)
    axes[1, 1].set_title(f"Diff: Basemodel - Target (MAE: {np.mean(np.abs(diff_bm_target)):.2f})")
    axes[1, 1].axis('off'); fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(diff_s2_bm, cmap=diff_cmap, vmin=-max_abs_diff_val, vmax=max_abs_diff_val)
    axes[1, 2].set_title(f"Diff: Stage2 - Basemodel (MAE: {np.mean(np.abs(diff_s2_bm)):.2f})")
    axes[1, 2].axis('off'); fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_fig = os.path.join(save_dir, f"{prefix}_comparison_maps_{title_suffix}.png")
    plt.savefig(save_path_fig, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Stage2 comparison visualization to {save_path_fig}")

if __name__ == '__main__':
    logger.info(f"===== DDPM Stage 2 Training and Evaluation =====")
    logger.info(f"Full CONFIG: {json.dumps(CONFIG, indent=2)}") # 可以取消註解以查看完整配置

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
    basemodel_for_output_generation.eval() # eval() 應該在 load_state_dict 之後，並且如果模型已在正確設備上，則不需要再 .to(device)
    logger.info(f"Basemodel (for output generation, unified DDPM3D) 載入完成。")

    if 'norm_stats_flow' not in chkpt_basemodel_eval or 'sorted_flow_columns' not in chkpt_basemodel_eval:
        raise ValueError("Basemodel 檢查點必須包含 'norm_stats_flow' 和 'sorted_flow_columns'。")
    basemodel_norm_stats_source = chkpt_basemodel_eval['norm_stats_flow']
    basemodel_sorted_flow_cols_source = chkpt_basemodel_eval['sorted_flow_columns'] # Stage2Dataset 會用到
    CONFIG["cached_basemodel_mean"] = float(basemodel_norm_stats_source['mean'])
    CONFIG["cached_basemodel_std"] = float(basemodel_norm_stats_source['std'])
    if CONFIG["cached_basemodel_std"] < 1e-6: CONFIG["cached_basemodel_std"] = 1.0
    CONFIG["cached_basemodel_sorted_flow_columns"] = basemodel_sorted_flow_cols_source
    CONFIG["cached_basemodel_selected_sensor_info"] = chkpt_basemodel_eval.get('selected_sensor_info')
    CONFIG["cached_basemodel_grid_idx_to_rc_map"] = chkpt_basemodel_eval.get('grid_idx_to_rc_map')
    # 添加檢查確保這些信息確實被加載了
    if not CONFIG.get("cached_basemodel_selected_sensor_info") or \
    not CONFIG.get("cached_basemodel_grid_idx_to_rc_map") or \
    not CONFIG.get("cached_basemodel_sorted_flow_columns"):
        raise ValueError("Basemodel 檢查點缺少必要的網格映射資訊 (selected_sensor_info, grid_idx_to_rc_map, or sorted_flow_columns)。無法繼續。")
    else:
        logger.info("成功從 Basemodel 檢查點加載網格映射資訊到 CONFIG。")


# --- 步驟 2: 準備 Stage2 數據 ---
NEW_COND_FEATURE_COL = CONFIG["stage2_new_condition_feature_column"]
NEW_COND_OPERATOR = CONFIG["stage2_new_conditional_operator"]
NEW_COND_VALUE = CONFIG["stage2_new_conditional_value"]
STAGE2_MODEL_NAME = CONFIG["stage2_model_name"]


logger.info(f"===== STAGE 2: 數據準備 =====")
df_for_stage2_processing = full_df.copy() # 或者您需要的其他基礎數據集
logger.info(f"Stage2: 使用數據 {len(df_for_stage2_processing)} 行進行處理。")
# NEW_COND_FEATURE_COL, NEW_COND_OPERATOR, NEW_COND_VALUE 仍然可以用於 Stage2Dataset 內部生成分類

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

# --- 新增：對 Basemodel 的輸出 (作為 Stage2 條件1) 進行標準化 ---
# 計算 all_bm_outputs_s2_np_cond 的均值和標準差
# 注意：這裡的統計量是基於篩選後的 Stage2 數據對應的 Basemodel 輸出來計算的
bm_cond_mean_for_s2 = np.mean(all_bm_outputs_s2_np_cond)
bm_cond_std_for_s2 = np.std(all_bm_outputs_s2_np_cond)
if bm_cond_std_for_s2 < 1e-6: # 避免除以非常小的數
    logger.warning(f"Basemodel 輸出條件的標準差過小 ({bm_cond_std_for_s2})，將設為 1.0 以避免除零錯誤。")
    bm_cond_std_for_s2 = 1.0

logger.info(f"將用於 Stage2 條件1 (Basemodel輸出) 的正規化統計: Mean={bm_cond_mean_for_s2:.4f}, Std={bm_cond_std_for_s2:.4f}")

# 進行標準化
all_bm_outputs_s2_np_cond_normalized = (all_bm_outputs_s2_np_cond - bm_cond_mean_for_s2) / bm_cond_std_for_s2
logger.info(f"正規化後的 Basemodel 輸出 (作為 Stage2 條件1) 的統計: Mean={np.mean(all_bm_outputs_s2_np_cond_normalized):.4f}, Std={np.std(all_bm_outputs_s2_np_cond_normalized):.4f}, Min={np.min(all_bm_outputs_s2_np_cond_normalized):.4f}, Max={np.max(all_bm_outputs_s2_np_cond_normalized):.4f}")

# 更新 CONFIG 以便 Stage2Dataset 如果需要可以訪問這些統計量（雖然目前 Dataset 是直接接收處理後的 numpy array）
CONFIG["cached_s2_cond1_norm_mean"] = bm_cond_mean_for_s2
CONFIG["cached_s2_cond1_norm_std"] = bm_cond_std_for_s2
# --- 標準化結束 ---

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
config_for_s2_dataset_use["new_conditional_feature_column"] = NEW_COND_FEATURE_COL
config_for_s2_dataset_use["new_conditional_operator"] = NEW_COND_OPERATOR
config_for_s2_dataset_use["new_conditional_value"] = NEW_COND_VALUE

train_dataset_s2 = Stage2Dataset(
    df_for_stage2_targets_and_conditions=df_for_stage2_processing.iloc[s2_train_indices_final],
    # 使用正規化後的 Basemodel 輸出
    basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_train_indices_final],
    config=config_for_s2_dataset_use, mode='train',
    original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
    # new_cond_feature_norm_stats_from_train 在訓練模式下由 Dataset 內部計算
)
s2_train_batch_size = CONFIG.get("batch_size")
train_loader_s2 = DataLoader(train_dataset_s2, batch_size=s2_train_batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True if len(train_dataset_s2) >= s2_train_batch_size else False)

val_loader_s2 = None
if len(s2_val_indices_final) > 0:
    val_dataset_s2 = Stage2Dataset(
        df_for_stage2_targets_and_conditions=df_for_stage2_processing.iloc[s2_val_indices_final],
        # 使用正規化後的 Basemodel 輸出
        basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_val_indices_final],
        config=config_for_s2_dataset_use, mode='val',
        stage2_target_stats_from_train=train_dataset_s2.norm_stats_target_s2,
        stage2_avg_flow_map_dict_from_train=train_dataset_s2.average_flow_map_dict_s2,
        original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
        new_cond_feature_norm_stats_from_train=train_dataset_s2.norm_stats_new_cond_feature # 這個是紫外線指數的正規化參數
    )
    s2_eval_batch_size = CONFIG.get("eval_batch_size") 
    val_loader_s2 = DataLoader(val_dataset_s2, batch_size=s2_eval_batch_size, shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    logger.info(f"Stage2 驗證數據集創建，含 {len(val_dataset_s2)} 樣本。")
else:
    logger.info("Stage2 驗證集為空。")

# --- 步驟 5: 訓練 Stage2 模型 ---
optimizer_s2 = optim.AdamW(list(stage2_model.parameters()), lr=CONFIG["lr_stage2"], weight_decay=CONFIG["weight_decay"])
scheduler_factor_s2 = CONFIG.get("lr_scheduler_factor")
scheduler_patience_s2 = CONFIG.get("lr_scheduler_patience")
scheduler_min_lr_s2 = CONFIG.get("lr_scheduler_min_lr")
# (可以再為 early_stopping_patience 添加類似的邏輯)
early_stopping_patience_s2 = CONFIG.get("early_stopping_patience")


scheduler_s2 = ReduceLROnPlateau(optimizer_s2,
                                 mode='min',
                                 factor=scheduler_factor_s2,
                                 patience=scheduler_patience_s2,
                                 min_lr=scheduler_min_lr_s2)

start_epoch_s2_train = 1
best_val_loss_s2_train = float('inf')
early_stopping_counter_s2_train = 0
metrics_hist_s2_train = {'train_loss':[], 'val_loss':[], 'lr':[]}
stage2_model_save_checkpoint_path = os.path.join(CONFIG["stage2_checkpoint_path"])

if CONFIG["resume_from_stage2_checkpoint"] and os.path.exists(stage2_model_save_checkpoint_path):
    logger.info(f"從 Stage2 檢查點恢復訓練: {stage2_model_save_checkpoint_path}")
    chkpt_s2_resume = torch.load(stage2_model_save_checkpoint_path, map_location=CONFIG["device"])
    stage2_model.load_state_dict(chkpt_s2_resume['ddpm_state_dict'])
    optimizer_s2.load_state_dict(chkpt_s2_resume['optimizer_state_dict'])
    if 'scheduler_state_dict' in chkpt_s2_resume: scheduler_s2.load_state_dict(chkpt_s2_resume['scheduler_state_dict'])
    start_epoch_s2_train = chkpt_s2_resume.get('epoch', 0) + 1
    best_val_loss_s2_train = chkpt_s2_resume.get('best_val_loss_s2', float('inf'))
    early_stopping_counter_s2_train = chkpt_s2_resume.get('early_stopping_counter_s2',0)
    metrics_hist_s2_train = chkpt_s2_resume.get('metrics_hist_s2', metrics_hist_s2_train)
    logger.info(f"Stage2 訓練將從 epoch {start_epoch_s2_train} 開始。")

logger.info(f"開始訓練 Stage2 模型: {STAGE2_MODEL_NAME}...")
epochs_to_run_s2 = CONFIG["epochs_stage2"]
for epoch_s2_current in range(start_epoch_s2_train, epochs_to_run_s2 + 1):
    stage2_model.train()
    total_train_loss_epoch_s2 = 0
    train_pbar_s2_loop = tqdm(train_loader_s2, desc=f"Stage2 Epoch {epoch_s2_current}/{epochs_to_run_s2} [Train]", leave=False)
    for target_s2_b, bm_out_grid_b, uv_grid_b, _, _ in train_pbar_s2_loop:
        optimizer_s2.zero_grad()
        target_s2_b = target_s2_b.to(CONFIG["device"])
        bm_out_grid_b = bm_out_grid_b.to(CONFIG["device"])
        uv_grid_b = uv_grid_b.to(CONFIG["device"])
        
        t_s2_b = torch.randint(0, stage2_model.timesteps, (target_s2_b.shape[0],), device=CONFIG["device"]).long()
        loss_val_s2 = stage2_model.p_losses(
            x_start_target_flow=target_s2_b,
            t=t_s2_b,
            basemodel_output_grid_batch=bm_out_grid_b,  # 明確指定
            new_condition_feature_grid_batch=uv_grid_b  # 明確指定
        )
        loss_val_s2.backward()
        optimizer_s2.step()
        total_train_loss_epoch_s2 += loss_val_s2.item()
        train_pbar_s2_loop.set_postfix({"S2 Loss": loss_val_s2.item()})
    
    avg_train_loss_epoch_s2 = total_train_loss_epoch_s2 / len(train_loader_s2)
    metrics_hist_s2_train['train_loss'].append(avg_train_loss_epoch_s2)
    current_lr_epoch_s2 = optimizer_s2.param_groups[0]['lr']
    metrics_hist_s2_train['lr'].append(current_lr_epoch_s2)

    avg_val_loss_s2_calculated_epoch = float('inf')
    val_calculated_this_epoch = False
    if val_loader_s2 and (epoch_s2_current % CONFIG.get("val_calculation_freq_stage2", 1) == 0 or epoch_s2_current == 1 or epoch_s2_current == epochs_to_run_s2):
        val_calculated_this_epoch = True
        stage2_model.eval()
        total_val_loss_p_s2 = 0
        num_val_samples_p_s2 = 0
        with torch.no_grad():
            for target_s2_v_b, bm_out_v_b, uv_v_b, _, _ in val_loader_s2:
                target_s2_v_b = target_s2_v_b.to(CONFIG["device"])
                bm_out_v_b = bm_out_v_b.to(CONFIG["device"])
                uv_v_b = uv_v_b.to(CONFIG["device"])
                t_s2_v_b = torch.randint(0, stage2_model.timesteps, (target_s2_v_b.shape[0],), device=CONFIG["device"]).long()
                val_loss_b_s2 = stage2_model.p_losses(
                    x_start_target_flow=target_s2_v_b,
                    t=t_s2_v_b,
                    basemodel_output_grid_batch=bm_out_v_b,  # 明確指定
                    new_condition_feature_grid_batch=uv_v_b  # 明確指定
                )
                total_val_loss_p_s2 += val_loss_b_s2.item() * target_s2_v_b.shape[0]
                num_val_samples_p_s2 += target_s2_v_b.shape[0]
        if num_val_samples_p_s2 > 0: avg_val_loss_s2_calculated_epoch = total_val_loss_p_s2 / num_val_samples_p_s2
        scheduler_s2.step(avg_val_loss_s2_calculated_epoch)

        if avg_val_loss_s2_calculated_epoch < best_val_loss_s2_train:
            best_val_loss_s2_train = avg_val_loss_s2_calculated_epoch
            early_stopping_counter_s2_train = 0
            torch.save({
                'epoch': epoch_s2_current,
                'ddpm_state_dict': stage2_model.state_dict(),
                'optimizer_state_dict': optimizer_s2.state_dict(),
                'scheduler_state_dict': scheduler_s2.state_dict(),
                'best_val_loss_s2': best_val_loss_s2_train,
                'config_snapshot_at_save': config_for_s2_dataset_use,
                'metrics_hist_s2': metrics_hist_s2_train,
                'early_stopping_counter_s2': early_stopping_counter_s2_train,
                'stage2_target_norm_stats': train_dataset_s2.norm_stats_target_s2,
                'stage2_avg_flow_map_dict': train_dataset_s2.average_flow_map_dict_s2,
                # 新增：保存新條件特徵的正規化統計量
                'new_cond_feature_norm_stats': train_dataset_s2.norm_stats_new_cond_feature
            }, stage2_model_save_checkpoint_path)
            logger.info(f"Stage2 Epoch {epoch_s2_current}: 新最佳模型已儲存 (Val Loss: {best_val_loss_s2_train:.5f})。")
        else:
            early_stopping_counter_s2_train += 1
    
    metrics_hist_s2_train['val_loss'].append(avg_val_loss_s2_calculated_epoch if val_calculated_this_epoch else metrics_hist_s2_train['val_loss'][-1] if metrics_hist_s2_train['val_loss'] else float('inf') )
    val_loss_display_s2 = f"{metrics_hist_s2_train['val_loss'][-1]:.5f}" if metrics_hist_s2_train['val_loss'][-1] != float('inf') else "N/A"
    logger.info(f"Stage2 Epoch {epoch_s2_current}: Train Loss: {avg_train_loss_epoch_s2:.5f} | Val Loss: {val_loss_display_s2} | LR: {current_lr_epoch_s2:.8f}")

    if early_stopping_counter_s2_train >= CONFIG["early_stopping_patience"]:
        logger.info(f"Stage2 訓練因早停機制觸發於 Epoch {epoch_s2_current}。")
        break
logger.info(f"Stage2 模型 '{STAGE2_MODEL_NAME}' 訓練完成。")

# --- Stage2 模型最終評估 ---
logger.info(f"===== STAGE 2: 最終模型評估 ({STAGE2_MODEL_NAME}) =====")
path_to_load_best_s2_model = CONFIG["stage2_checkpoint_full_path"]
if not os.path.exists(path_to_load_best_s2_model):
    logger.warning(f"找不到最佳 Stage2 模型檔案: {path_to_load_best_s2_model}。將使用訓練結束時的 Stage2 模型狀態進行評估。")
    final_s2_model_for_eval_load = stage2_model
    chkpt_s2_final_for_eval = {'epoch': epochs_to_run_s2 } # 模擬一個檢查點字典
    # 嘗試從 train_dataset_s2 獲取統計數據，如果模型是訓練結束時的狀態
    s2_target_stats_for_final_eval = train_dataset_s2.norm_stats_target_s2 if hasattr(train_dataset_s2, 'norm_stats_target_s2') else None
    s2_avg_flow_map_for_final_eval = train_dataset_s2.average_flow_map_dict_s2 if hasattr(train_dataset_s2, 'average_flow_map_dict_s2') else None
else:
    logger.info(f"從 {path_to_load_best_s2_model} 載入最佳 Stage2 模型進行評估...")
    chkpt_s2_final_for_eval = torch.load(path_to_load_best_s2_model, map_location=CONFIG["device"], weights_only=False)
    
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
    s2_target_stats_for_final_eval = chkpt_s2_final_for_eval.get('stage2_target_norm_stats')
    s2_avg_flow_map_for_final_eval = chkpt_s2_final_for_eval.get('stage2_avg_flow_map_dict')
    new_cond_feature_norm_stats_for_final_eval = chkpt_s2_final_for_eval.get('new_cond_feature_norm_stats')

if s2_target_stats_for_final_eval is None or s2_avg_flow_map_for_final_eval is None or new_cond_feature_norm_stats_for_final_eval is None: # 修改判斷
        # 如果是從訓練結束時的狀態（而不是檢查點）獲取，則嘗試從 train_dataset_s2 獲取
        if not os.path.exists(path_to_load_best_s2_model): # 表示使用的是訓練結束時的狀態
             if hasattr(train_dataset_s2, 'norm_stats_target_s2'): s2_target_stats_for_final_eval = train_dataset_s2.norm_stats_target_s2
             if hasattr(train_dataset_s2, 'average_flow_map_dict_s2'): s2_avg_flow_map_for_final_eval = train_dataset_s2.average_flow_map_dict_s2
             if hasattr(train_dataset_s2, 'norm_stats_new_cond_feature'): new_cond_feature_norm_stats_for_final_eval = train_dataset_s2.norm_stats_new_cond_feature

        if s2_target_stats_for_final_eval is None or s2_avg_flow_map_for_final_eval is None or new_cond_feature_norm_stats_for_final_eval is None:
             raise ValueError("無法為最終評估獲取 Stage2 目標或新條件的正規化統計量或平均流量圖字典。")

# 準備測試集 Loader
test_loader_s2_final = None
if len(s2_test_indices_final) > 0:
    test_dataset_s2_final = Stage2Dataset(
        df_for_stage2_targets_and_conditions=df_for_stage2_processing.iloc[s2_test_indices_final],
        # 使用正規化後的 Basemodel 輸出
        basemodel_outputs_for_samples_np=all_bm_outputs_s2_np_cond_normalized[s2_test_indices_final],
        config=config_for_s2_dataset_use, mode='test',
        stage2_target_stats_from_train=s2_target_stats_for_final_eval,
        stage2_avg_flow_map_dict_from_train=s2_avg_flow_map_for_final_eval,
        original_sorted_flow_columns=basemodel_sorted_flow_cols_source,
        new_cond_feature_norm_stats_from_train=new_cond_feature_norm_stats_for_final_eval # 這個是紫外線指數的正規化參數
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

    if excel_rows_final_s2:
        df_excel_final_s2 = pd.DataFrame(excel_rows_final_s2)
        excel_column_order_s2 = ['資料來源', '網格座標_R', '網格座標_C', '經度', '緯度', 'MSE', 'MAE', 'MAPE', 'SMAPE', 'FID']
        # 重新排序列，如果某列不存在，則會產生 KeyError，所以要小心
        df_excel_final_s2 = df_excel_final_s2.reindex(columns=excel_column_order_s2)

        excel_final_path_s2 = os.path.join(CONFIG["stage2_model_save_dir"], f"final_test_metrics_detailed_{STAGE2_MODEL_NAME}.xlsx")
        df_excel_final_s2.to_excel(excel_final_path_s2, index=False)
        logger.info(f"Stage2 詳細測試評估指標已匯出至: {excel_final_path_s2}")

else:
    logger.warning("Stage2 最終評估的測試數據集為空，跳過評估。")

logger.info(f"===== Stage2 流程全部結束 ({STAGE2_MODEL_NAME}) =====")
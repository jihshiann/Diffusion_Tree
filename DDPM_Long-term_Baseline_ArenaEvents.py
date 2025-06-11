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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from typing import Optional, Tuple, List, Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from enum import Enum

# ==============================================================================
# 組態設定 (專為 5-Channel Baseline 模型設計)
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    # --- 資料參數 ---
    "data_path": r"C:\thesis\code\Taipei_CF\all_merged.csv",
    "H": 20,
    "W": 20,
    "D": 1,
    # 這是從舊的 Basemodel 檢查點載入的資訊，Baseline模型也需要它來解析流量欄位
    "basemodel_checkpoint": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_long-term\best_ddpm_model_during_training.pth",

    # --- 模型架構參數 ---
    "image_channels": 1,      # 主要資料(流量圖)的通道數
    "base_channels_unet": 64,   # UNet3D 的基礎通道數
    "unet_dropout_rate": 0.1,
    "time_emb_dim": 256,        # 時間嵌入維度
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 / UNet中與x_t合併的維度
    
    # --- Baseline 模型要使用的 5 個條件特徵 ---
    "baseline_feature_columns": [
        "時", 
        "月", 
        "日", 
        "holiday", 
        "month_day_combined"
    ],
    # 模型的條件處理器現在需要接收 5 個通道
    "condition_input_channels": 5, 
    
    # === Baseline 專家模型配置 ===
    "model_name": "Baseline_ArenaEvents",
    "checkpoint_path": "best_baseline_model_arenaEvents.pth",

    # === Stage2 特定配置 ===
    "stage2_new_condition_feature_column": "時", # Stage2 新條件的欄位名
    "stage2_new_conditional_operator": "<=",         # Stage2 新條件的運算符
    "stage2_new_conditional_value": 20,             # Stage2 新條件的閾值
    "stage2_model_name": "stage2_HourLe20",    # 第二階段模型的名稱
    "stage2_checkpoint_path": "best_stage2_model_hour_le_20.pth", # Stage2 模型的檢查點檔名 (相對路徑)

    #hgt     === Stage3 特定配置 ===
    "stage3_new_condition_feature_column": "露點溫度", # Stage3 新條件的欄位名 
    "stage3_new_conditional_operator": "<=",         # Stage3 新條件的運算符
    "stage3_new_conditional_value": 23.5,             # Stage3 新條件的閾值
    "stage3_model_name": "Stage3_DewPointLe235",    # 第三階段模型的名稱
    "stage3_checkpoint_path": "best_stage3_model_DewPoint_le_23_5.pth", # Stage3 模型的檢查點檔名 (相對路徑)

    # === 過濾規則：基於外部 Excel 檔案 ===
    "stage4_config": {
        "model_name": "stage4_ArenaEventDays_CombinedDateCond",
        "checkpoint_path": "best_stage4_model_arena_events_date_cond.pth",
        
        # 1. 定義基於事件的過濾規則
        "event_filter": {
            "enabled": True,
            "file_path": r"C:\thesis\code\Taipei_CF\ArenaEvents.xlsx",
            "month_col": "月",
            "day_col": "日"
        },
        
        # 2. 指定用於【模型條件輸入】的組合特徵欄位名稱
        "grid_feature_source_column": "month_day_combined"
    },
    
    # --- DDPM 擴散參數 ---
    "timesteps": 1000,
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
    "val_calculation_freq": 2,

    # --- 評估參數 ---
    "eval_batch_size": 128,
    "fid_batch_size": 64,
    "fid_num_samples": 128,
    "mape_threshold": 1.0,

    # --- 路徑與儲存 ---
    "save_dir": "results_ddpm_baseline", # 新的結果儲存目錄
    
    "train_split_ratio": 0.7,
    "val_split_ratio": 0.15,
}

# --- 路徑生成 ---
CONFIG["model_save_dir"] = os.path.join(CONFIG["save_dir"], CONFIG["model_name"])
os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
CONFIG["checkpoint_full_path"] = os.path.join(CONFIG["model_save_dir"], CONFIG["checkpoint_path"])

# --- 初始化隨機種子 ---
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])

logger.info(f"使用裝置: {CONFIG['device']}")
logger.info(f"Baseline 模型結果將儲存於: {CONFIG['model_save_dir']}")


# --- 輔助函式與枚舉 ---

def create_condition_mask(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
    """根據指定的條件，為 DataFrame 創建一個布林遮罩。"""
    if column not in df.columns:
        raise ValueError(f"欄位 '{column}' 不存在於 DataFrame 中。")
    series_vals = pd.to_numeric(df[column], errors='coerce')
    
    if operator == "<=": mask = (series_vals <= float(value))
    elif operator == ">": mask = (series_vals > float(value))
    elif operator == "<": mask = (series_vals < float(value))
    elif operator == ">=": mask = (series_vals >= float(value))
    elif operator == "==": mask = (series_vals == float(value))
    elif operator == "!=": mask = (series_vals != float(value))
    else: raise ValueError(f"不支援的運算符: '{operator}'")

    return mask.fillna(False)

class ConditionMode(Enum):
    # 為 Baseline 模型定義一個模式，方便未來擴展或調試
    BASELINE = 1

#%%
# ==============================================================================
# UNet3D, DDPM3D 模型定義
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦時間嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DoubleConv3D(nn.Module):
    """(卷積3D -> BN -> SiLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down3D(nn.Module):
    """下採樣模組 (MaxPool3D -> DoubleConv3D)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), # 深度維度不壓縮
            DoubleConv3D(in_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """上採樣模組"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True) # 深度維度不放大
            self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """輸出卷積層 (1x1x1 Conv3D)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net 模型"""
    def __init__(self, input_image_channels: int, base_channels: int, time_emb_dim: int,
                 condition_encode_dim: Optional[int] = None, bilinear_upsample: bool = True, dropout_rate: float = 0.05):
        super().__init__()
        self.input_image_channels = input_image_channels
        self.condition_encode_dim = condition_encode_dim or 0
        actual_in_channels = self.input_image_channels + self.condition_encode_dim
        
        self.shared_time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.inc = DoubleConv3D(actual_in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear_upsample else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, bilinear_upsample)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, bilinear_upsample)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, bilinear_upsample)
        self.up4 = Up3D(base_channels * 2, base_channels, bilinear_upsample)
        self.outc = OutConv3D(base_channels, self.input_image_channels)

        time_proj_dims = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16//factor,
                          base_channels*8//factor, base_channels*4//factor, base_channels*2//factor, base_channels]
        time_proj_names = ['inc', 'down1', 'down2', 'down3', 'bottleneck', 'up1', 'up2', 'up3', 'up4']
        for name, dim in zip(time_proj_names, time_proj_dims):
            setattr(self, f"time_proj_{name}", nn.Linear(time_emb_dim, dim))

    def _add_time_embedding(self, x, t_emb, proj_layer):
        t_emb_projected = proj_layer(t_emb)
        return x + t_emb_projected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x_t, time_steps, processed_condition=None):
        shared_t_emb = self.shared_time_mlp(time_steps)
        x_input = torch.cat((x_t, processed_condition), dim=1) if processed_condition is not None else x_t
        
        x1 = self._add_time_embedding(self.inc(x_input), shared_t_emb, self.time_proj_inc)
        x2 = self._add_time_embedding(self.down1(x1), shared_t_emb, self.time_proj_down1)
        x3 = self._add_time_embedding(self.down2(x2), shared_t_emb, self.time_proj_down2)
        x4 = self._add_time_embedding(self.down3(x3), shared_t_emb, self.time_proj_down3)
        x5 = self._add_time_embedding(self.down4(x4), shared_t_emb, self.time_proj_bottleneck)
        x5 = self.dropout(x5)

        x = self._add_time_embedding(self.up1(x5, x4), shared_t_emb, self.time_proj_up1)
        x = self._add_time_embedding(self.up2(x, x3), shared_t_emb, self.time_proj_up2)
        x = self._add_time_embedding(self.up3(x, x2), shared_t_emb, self.time_proj_up3)
        x = self._add_time_embedding(self.up4(x, x1), shared_t_emb, self.time_proj_up4)
        return self.outc(x)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM3D(nn.Module):
    def __init__(self, unet_model, timesteps, image_size, image_channels,
                 condition_input_channels, condition_encode_dim,
                 beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps
        self.image_size_D, self.image_size_H, self.image_size_W = image_size
        self.image_channels = image_channels
        self.device = device
        
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.condition_processor = nn.Sequential(
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()
        ).to(device)
        logger.info(f"DDPM3D instance created. Condition processor expects {condition_input_channels} input channels.")

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sact = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        soma_ct = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sact * x_start + soma_ct * noise

    def p_losses(self, x_start_target_flow, t, condition_grids, noise=None):
        if noise is None: noise = torch.randn_like(x_start_target_flow)
        x_t_noisy_target = self.q_sample(x_start=x_start_target_flow, t=t, noise=noise)
        
        # 直接處理傳入的多通道條件網格
        processed_condition = self.condition_processor(condition_grids.to(self.device))
        predicted_noise = self.model(x_t_noisy_target, t, processed_condition)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, batch_size, condition_grids):
        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)
        
        processed_conditions = self.condition_processor(condition_grids.to(self.device))

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            betas_t = self._extract(self.betas, t, img.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, img.shape)
            
            predicted_noise = self.model(img, t, processed_conditions)
            
            model_mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            if i == 0:
                img = model_mean
            else:
                posterior_variance_t = self._extract(self.posterior_variance, t, img.shape)
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise
        return img
    
#%%
# ==============================================================================
# 數據處理與評估
# ==============================================================================
def parse_lat_lon(column_name: str) -> tuple[float, float]:
    """從欄位名稱中解析經緯度。"""
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class BaselineDataset(Dataset):
    """為單階段 Baseline 模型設計的數據集類別。"""
    def __init__(self, df_for_processing, config, mode='train', 
                 norm_stats_from_train=None, target_info_from_train=None):
        super().__init__()
        self.df_processed = df_for_processing.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.BaselineDataset[{self.mode}]")

        self.H = config["H"]
        self.W = config["W"]
        self.D = config.get("D", 1)
        self.image_channels_target = config["image_channels"]
        
        # Basemodel 的網格資訊依然是共用的
        # 在主流程中，這些資訊應在初始化 Dataset 前被載入到 CONFIG 中
        self.sorted_flow_columns = config["cached_basemodel_sorted_flow_columns"]

        # --- 處理條件特徵 ---
        self._process_conditions(norm_stats_from_train)

        # --- 處理目標 (Target) ---
        self._calculate_or_load_targets(target_info_from_train)
        
        self.logger.info(f"BaselineDataset __init__ (mode={self.mode}) COMPLETED.")

    def _get_original_cond_values(self, col_name: str) -> np.ndarray:
        if col_name not in self.df_processed.columns:
            raise ValueError(f"Dataset: 條件欄位 '{col_name}' 不在 DataFrame 中。")
        return pd.to_numeric(self.df_processed[col_name], errors='coerce').values

    def _calculate_norm_stats(self, values_np: np.ndarray, col_name: str) -> Dict[str, float]:
        valid_values = values_np[~np.isnan(values_np)]
        mean_val, std_val = (np.mean(valid_values), np.std(valid_values)) if len(valid_values) > 0 else (0.0, 1.0)
        if std_val < 1e-6:
            self.logger.warning(f"特徵 '{col_name}' 的標準差過小，將其設為 1.0 以避免除以零。")
            std_val = 1.0
        return {'mean': mean_val, 'std': std_val}
        
    def _process_conditions(self, norm_stats_from_train=None):
        """處理 CONFIG 中定義的所有 baseline_feature_columns。"""
        self.feature_columns = self.config.get("baseline_feature_columns", [])
        if not self.feature_columns or len(self.feature_columns) != self.config["condition_input_channels"]:
            raise ValueError("CONFIG 中的 'baseline_feature_columns' 未定義或長度與 'condition_input_channels' 不符。")

        self.original_values_dict = {col: self._get_original_cond_values(col) for col in self.feature_columns}
        
        if self.mode == 'train':
            self.norm_stats_dict = {}
            self.logger.info("在訓練模式下，為所有條件特徵計算正規化統計量...")
            for col_name, values in self.original_values_dict.items():
                self.norm_stats_dict[col_name] = self._calculate_norm_stats(values, col_name)
                stats = self.norm_stats_dict[col_name]
                self.logger.info(f"  - 特徵 '{col_name}': mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        else: # val or test
            if norm_stats_from_train is None:
                raise ValueError("驗證/測試模式需要從訓練集傳入正規化統計量 (norm_stats_from_train)。")
            self.norm_stats_dict = norm_stats_from_train
            self.logger.info("在驗證/測試模式下，已載入訓練集的正規化統計量。")

    def _calculate_or_load_targets(self, target_info_from_train=None):
        """計算或載入目標流量圖及其正規化參數，並產生分組說明的日誌。"""
        
        # [新增] 初始化一個字典來存放分組條件的說明
        self.grouping_key_descriptions = {}

        # 1. 處理 Basemodel 條件 (小時, 假日)
        self.hours_for_target_np = self.df_processed['時'].values.astype(int)
        self.hour_category_for_target_grouping_np = (self.hours_for_target_np > 8).astype(int)
        self.grouping_key_descriptions['hour_category'] = "0: 時<=8, 1: 時>8" # [新增] 說明

        self.is_holiday_for_target_np = self.df_processed['holiday'].astype(bool).astype(int).values
        self.grouping_key_descriptions['is_holiday'] = "0: 非假日, 1: 假日" # [新增] 說明

        # 2. 處理 Stage2 條件
        s2_col = self.config["stage2_new_condition_feature_column"]
        s2_op = self.config["stage2_new_conditional_operator"]
        s2_val = self.config["stage2_new_conditional_value"]
        s2_vals_np = pd.to_numeric(self.df_processed[s2_col], errors='coerce').values
        # 邏輯是 ~(series <= value)，所以 1 代表 > value
        self.s2_cond_category_for_target_np = (~(pd.Series(s2_vals_np) <= s2_val)).astype(int)
        self.grouping_key_descriptions['s2_cond_category'] = f"0: {s2_col}{s2_op}{s2_val}, 1: NOT ({s2_col}{s2_op}{s2_val})" # [新增] 說明
        
        # 3. 處理 Stage3 條件
        s3_col = self.config["stage3_new_condition_feature_column"]
        s3_op = self.config["stage3_new_conditional_operator"]
        s3_val = self.config["stage3_new_conditional_value"]
        s3_vals_np = pd.to_numeric(self.df_processed[s3_col], errors='coerce').values
        self.s3_cond_category_for_target_np = (~(pd.Series(s3_vals_np) <= s3_val)).astype(int)
        self.grouping_key_descriptions['s3_cond_category'] = f"0: {s3_col}{s3_op}{s3_val}, 1: NOT ({s3_col}{s3_op}{s3_val})" # [新增] 說明

        # 4. 處理 Stage4/Baseline 自身條件
        self.s4_cond_category_for_target_np = np.zeros(len(self.df_processed), dtype=int)
        self.grouping_key_descriptions['s4_cond_category'] = "0: 專家模型(小巨蛋活動日)" # [新增] 說明
        
        # --- 後續邏輯不變 ---
        if self.mode == 'train':
            self.average_flow_map_dict = self._calculate_target_flows()
            all_maps = np.array(list(self.average_flow_map_dict.values())) if self.average_flow_map_dict else np.array([])
            self.norm_stats_target = self._calculate_norm_stats(all_maps.flatten(), "TargetFlows")
            self.logger.info(f"計算得到目標流量的專用正規化統計量: mean={self.norm_stats_target['mean']:.4f}, std={self.norm_stats_target['std']:.4f}")
        else:
            if target_info_from_train is None:
                raise ValueError("驗證/測試模式需要從訓練集傳入目標資訊 (target_info_from_train)。")
            self.average_flow_map_dict = target_info_from_train["avg_flow_map"]
            self.norm_stats_target = target_info_from_train["norm_stats"]
            self.logger.info("已載入目標流量的預計算平均圖和專用正規化統計量。")
            
    def _calculate_target_flows(self) -> Dict[Tuple, np.ndarray]:
        """基於所有複合條件分組，計算平均流量圖。"""
        flow_data = self.df_processed[self.sorted_flow_columns].values.astype(np.float32)
        
        grouping_df = pd.DataFrame({
            'hour_category': self.hour_category_for_target_grouping_np,
            'is_holiday': self.is_holiday_for_target_np,
            's2_cond_category': self.s2_cond_category_for_target_np,
            's3_cond_category': self.s3_cond_category_for_target_np,
            's4_cond_category': self.s4_cond_category_for_target_np
        })
        
        grouped = grouping_df.groupby(list(grouping_df.columns), observed=False)
        
        # <<< [修改] 這裡加入了更詳細的日誌記錄 >>>
        self.logger.info("===== 開始計算目標平均流量圖的分組詳情 =====")
        self.logger.info("分組依據 (Group Keys) 說明:")
        grouping_columns = list(grouping_df.columns)
        # 遍歷每個分組欄位，並印出它的說明
        for col_name in grouping_columns:
            description = self.grouping_key_descriptions.get(col_name, "無可用說明")
            self.logger.info(f"  - {col_name}: {description}")
        self.logger.info("-------------------------------------------------")


        avg_flows = {}
        for group_key, group_indices in grouped.indices.items():
            self.logger.info(f"  - 找到分組: {group_key}, 包含 {len(group_indices)} 筆資料")
            if len(group_indices) > 0:
                mean_flow_flat = np.nanmean(flow_data[group_indices], axis=0)
                mean_flow_flat[np.isnan(mean_flow_flat)] = 0
                avg_flows[group_key] = mean_flow_flat.reshape(self.H, self.W)

        self.logger.info("===== 分組詳情記錄結束 =====")
        self.logger.info(f"計算完成 {len(avg_flows)} 個複合條件的目標平均流量圖。")
        return avg_flows

    def __len__(self):
        return len(self.df_processed)

    def __getitem__(self, idx):
        # 1. 準備目標 (Target)
        target_key = (
            self.hour_category_for_target_grouping_np[idx], 
            self.is_holiday_for_target_np[idx],
            self.s2_cond_category_for_target_np[idx],
            self.s3_cond_category_for_target_np[idx],
            self.s4_cond_category_for_target_np[idx]
        )
        # 從字典中獲取對應的預計算平均流量圖
        target_avg_flow_np = self.average_flow_map_dict.get(target_key, np.zeros((self.H, self.W), dtype=np.float32))
        
        # 使用目標的專用正規化參數
        target_mean = self.norm_stats_target['mean']
        target_std = self.norm_stats_target['std']
        norm_target_np = (target_avg_flow_np - target_mean) / (target_std if target_std > 1e-6 else 1.0)
        target_tensor_norm = torch.from_numpy(norm_target_np).float().reshape(self.image_channels_target, self.D, self.H, self.W)

        # 2. 準備多通道條件張量
        condition_grids = []
        for col_name in self.feature_columns:
            original_value = self.original_values_dict[col_name][idx]
            norm_stats = self.norm_stats_dict[col_name]
            mean, std = norm_stats['mean'], norm_stats['std']
            
            normalized_value = (original_value - mean) / (std if std > 1e-6 else 1.0) if not np.isnan(original_value) else 0.0
            
            grid = torch.full((1, self.D, self.H, self.W), float(normalized_value), dtype=torch.float32)
            condition_grids.append(grid)
            
        condition_tensor_norm = torch.cat(condition_grids, dim=0) 

        return target_tensor_norm, condition_tensor_norm

# FID 函數
def get_activations(images: torch.Tensor, model: nn.Module, device: str, batch_size_fid: int) -> np.ndarray:
    """使用 Inception 模型提取影像特徵。"""
    model.eval()
    activations = []

    if images.shape[2] == 1:
        images_2d = images.squeeze(2)
    else:
        images_2d = images[:, :, images.shape[2]//2, :, :]
        logger.warning("影像深度 > 1，為 FID 取中間切片。")

    if images_2d.shape[1] == 1:
        images_2d = images_2d.repeat(1, 3, 1, 1)
    
    transform_inception = transforms.Compose([
        transforms.Resize((299, 299), antialias=True)
    ])

    num_batches = math.ceil(images_2d.shape[0] / batch_size_fid)
    for i in range(num_batches):
        batch = images_2d[i*batch_size_fid : (i+1)*batch_size_fid].to(device)
        batch = transform_inception(batch)
        with torch.no_grad():
            pred = model(batch)
        if isinstance(pred, tuple): pred = pred[0]
        activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)

def calculate_frechet_distance(mu1:np.ndarray, sigma1:np.ndarray, mu2:np.ndarray, sigma2:np.ndarray, eps:float=1e-6) -> float:
    """計算兩個多元高斯分佈之間的 Fréchet Distance。"""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean_sqrt, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean_sqrt).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean_sqrt = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean_sqrt):
        covmean_sqrt = covmean_sqrt.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean_sqrt)
def calculate_fid(real_acts:np.ndarray, gen_acts:np.ndarray)->float:
    """計算給定真實與生成影像特徵的 FID 分數。"""
    mu_real, sigma_real = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu_gen, sigma_gen = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

# 評估與視覺化函數
def visualize_predictions(
    generated_all_denorm_t: torch.Tensor,
    original_all_denorm_t: torch.Tensor,
    config: Dict[str, Any],
    sample_idx_to_plot: Optional[int] = 0,
    prefix: str = "eval"
):
    """
    視覺化預測結果與真實值的比較。
    包含生成結果、真實數據、以及誤差（MSE、MAE、MAPE、SMAPE）的網格熱力圖。
    """
    save_dir = config["model_save_dir"] # 使用 Baseline 的儲存路徑
    os.makedirs(save_dir, exist_ok=True)

    # 數據深度 D 始終為1，直接壓縮
    generated_squeezed = generated_all_denorm_t.squeeze(1).squeeze(1)
    original_squeezed = original_all_denorm_t.squeeze(1).squeeze(1)

    if sample_idx_to_plot is None:
        gen_data_to_plot = torch.mean(generated_squeezed, dim=0).cpu().numpy()
        orig_data_to_plot = torch.mean(original_squeezed, dim=0).cpu().numpy()
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

    epsilon = 1e-8
    mse_matrix = (gen_data_to_plot - orig_data_to_plot) ** 2
    mae_matrix = np.abs(gen_data_to_plot - orig_data_to_plot)
    mape_matrix = np.abs((orig_data_to_plot - gen_data_to_plot) / (np.abs(orig_data_to_plot) + epsilon)) * 100
    smape_matrix = np.abs(gen_data_to_plot - orig_data_to_plot) / ((np.abs(orig_data_to_plot) + np.abs(gen_data_to_plot))/2 + epsilon) * 100 

    overall_mse = np.mean(mse_matrix)
    overall_mae = np.mean(mae_matrix)
    overall_mape = np.mean(mape_matrix[np.isfinite(mape_matrix)]) if np.any(np.isfinite(mape_matrix)) else float('inf')
    overall_smape = np.mean(smape_matrix[np.isfinite(smape_matrix)]) if np.any(np.isfinite(smape_matrix)) else float('inf')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Comparison for {prefix}_{title_suffix}', fontsize=16)

    # 圖 1: Generated
    im_gen = axes[0, 0].imshow(gen_data_to_plot, cmap='viridis')
    axes[0, 0].set_title(f'Generated ({title_suffix})')
    axes[0, 0].axis('off')
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
    vmax_mape = np.percentile(mape_matrix[np.isfinite(mape_matrix)], 98) if np.any(np.isfinite(mape_matrix)) else 100
    im_mape = axes[1, 1].imshow(mape_matrix, cmap='cividis', vmin=0, vmax=vmax_mape if vmax_mape > 0 else 100)
    axes[1, 1].set_title(f'MAPE Grid (Avg: {overall_mape:.0f})')
    axes[1, 1].axis('off')
    fig.colorbar(im_mape, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 圖 6: SMAPE
    vmax_smape = np.percentile(smape_matrix[np.isfinite(smape_matrix)], 98) if np.any(np.isfinite(smape_matrix)) else 100
    im_smape = axes[1, 2].imshow(smape_matrix, cmap='cividis', vmin=0, vmax=vmax_smape if vmax_smape > 0 else 100)
    axes[1, 2].set_title(f'SMAPE Grid (Avg: {overall_smape:.0f})')
    axes[1, 2].axis('off')
    fig.colorbar(im_smape, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'{prefix}_6maps_comparison_{title_suffix}.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"已儲存比較圖: {save_path}")

def plot_grid_with_error(
    dataset_for_coords: Dataset,
    error_metrics_grids: Dict[str, np.ndarray],
    config: Dict[str, Any],
    prefix: str = "eval"
):
    """在地理網格上繪製誤差指標 (同步 Stage4 風格)。"""
    save_dir = config["model_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    H, W = config["H"], config["W"]
    
    sorted_flow_columns_map = config.get("cached_basemodel_sorted_flow_columns")
    selected_sensor_info_plot = config.get("cached_basemodel_selected_sensor_info")

    if not all([sorted_flow_columns_map, selected_sensor_info_plot]):
        logger.error("plot_grid_with_error: CONFIG 中缺少必要的網格映射資訊。")
        return

    selected_sensor_info_dict = {info['name']: (info['lon'], info['lat']) for info in selected_sensor_info_plot}
    
    actual_sensor_lons, actual_sensor_lats, valid_grid_indices_flat = [], [], []
    for flat_grid_idx in range(H * W):
        if flat_grid_idx < len(sorted_flow_columns_map):
            col_name = sorted_flow_columns_map[flat_grid_idx]
            if col_name in selected_sensor_info_dict:
                lon, lat = selected_sensor_info_dict[col_name]
                actual_sensor_lons.append(lon)
                actual_sensor_lats.append(lat)
                valid_grid_indices_flat.append(flat_grid_idx)
    
    if not actual_sensor_lons:
        logger.error("plot_grid_with_error: 無法獲取任何網格點的座標。")
        return

    cdict_red_to_black = {
        'red':   ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    red_to_black_cmap = mcolors.LinearSegmentedColormap('RedToBlack', cdict_red_to_black)

    for metric_name, error_grid_flat in error_metrics_grids.items():
        if not isinstance(error_grid_flat, np.ndarray) or error_grid_flat.size != H*W:
            logger.error(f"Metric {metric_name} 的誤差網格維度不正確。跳過繪圖。")
            continue
            
        error_values_for_plot = error_grid_flat[valid_grid_indices_flat]
        
        plt.figure(figsize=(12, 12))
        
        is_diff_plot = "Diff_" in metric_name
        if is_diff_plot:
            cmap = 'bwr' # 使用藍白紅漸層來顯示差異
            abs_max = np.nanmax(np.abs(error_values_for_plot))
            vmin, vmax = -abs_max, abs_max
        else:
            cmap = red_to_black_cmap
            vmin = np.nanmin(error_values_for_plot[np.isfinite(error_values_for_plot)]) if np.any(np.isfinite(error_values_for_plot)) else 0
            vmax = np.nanmax(error_values_for_plot[np.isfinite(error_values_for_plot)]) if np.any(np.isfinite(error_values_for_plot)) else 1

        scatter = plt.scatter(actual_sensor_lons, actual_sensor_lats, c=error_values_for_plot, 
                                cmap=cmap, marker='s', s=100, vmin=vmin, vmax=vmax)
        
        plt.colorbar(scatter, label=metric_name)
        
        if "MSE" not in metric_name.upper():
            for i, val in enumerate(error_values_for_plot):
                if np.isfinite(val):
                    plt.text(actual_sensor_lons[i], actual_sensor_lats[i], f'{val:.0f}',
                             fontsize=6, color='white', ha='center', va='center')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Geographic Grid Error Heatmap - {metric_name.upper()} ({prefix})")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box')
        save_path = os.path.join(save_dir, f'{prefix}_grid_{metric_name.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"已儲存地理誤差圖: {save_path}")
@torch.no_grad()
def evaluate_baseline_model(
    model_trained: 'DDPM3D',
    dataloader: DataLoader,
    inception_model_fid: nn.Module,
    config: Dict[str, Any],
    target_norm_stats: Dict[str, float],
    max_samples_for_fid: Optional[int] = None,
    prefix: str = "eval_baseline"
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    
    logger.info(f"===== 開始 Baseline 模型評估 ({prefix}) =====")
    model_trained.eval()
    inception_model_fid.eval()

    target_mean = target_norm_stats['mean']
    target_std = target_norm_stats['std'] if target_norm_stats['std'] > 1e-6 else 1.0
    epsilon = 1e-8

    all_generated_denorm, all_target_denorm = [], []
    all_generated_norm, all_target_norm = [], []
    
    # 收集 FID 樣本
    max_fid_samples_actual = len(dataloader.dataset)
    if max_samples_for_fid is not None:
        max_fid_samples_actual = min(max_samples_for_fid, max_fid_samples_actual)

    pbar_eval = tqdm(dataloader, desc=f"Baseline 評估 ({prefix})", leave=False)
    for target_norm_b, cond_norm_b in pbar_eval:
        target_norm = target_norm_b.to(config["device"])
        cond_norm = cond_norm_b.to(config["device"])
        
        generated_norm = model_trained.sample(target_norm.shape[0], cond_norm)
        
        generated_denorm = generated_norm * target_std + target_mean
        target_denorm = target_norm * target_std + target_mean
        generated_denorm = torch.clamp(generated_denorm, min=0.0)

        all_generated_denorm.append(generated_denorm.cpu())
        all_target_denorm.append(target_denorm.cpu())
        
        samples_collected_so_far = sum(s.shape[0] for s in all_target_norm)
        if samples_collected_so_far < max_fid_samples_actual:
            all_generated_norm.append(generated_norm.cpu())
            all_target_norm.append(target_norm.cpu())

    pred_t = torch.cat(all_generated_denorm, dim=0)
    target_t = torch.cat(all_target_denorm, dim=0)
    
    # --- 指標計算 (同步 Stage4 邏輯) ---
    mse = F.mse_loss(pred_t, target_t).item()
    mae = F.l1_loss(pred_t, target_t).item()

    # 計算 mape_avg_grid 和 smape_avg_grid
    actual_values_per_element = torch.abs(target_t)
    errors_per_element = torch.abs(target_t - pred_t)
    
    threshold = config.get("mape_threshold", 1.0)
    valid_mape_mask = actual_values_per_element > threshold
    
    mape_avg_grid = float('inf')
    if torch.sum(valid_mape_mask).item() > 0:
        mape_per_element_filtered = (errors_per_element[valid_mape_mask] / actual_values_per_element[valid_mape_mask]) * 100
        mape_per_element_finite = mape_per_element_filtered[torch.isfinite(mape_per_element_filtered)]
        if mape_per_element_finite.numel() > 0:
            mape_avg_grid = torch.mean(mape_per_element_finite).item()

    smape_numerator_per_element = errors_per_element
    smape_denominator_per_element = (actual_values_per_element + torch.abs(pred_t)) / 2.0 + epsilon
    smape_per_element = (smape_numerator_per_element / smape_denominator_per_element) * 100
    smape_per_element_finite = smape_per_element[torch.isfinite(smape_per_element)]
    smape_avg_grid = torch.mean(smape_per_element_finite).item() if smape_per_element_finite.numel() > 0 else float('inf')
    
    # 計算 mape_overall 和 smape_overall
    mape_overall = (torch.sum(errors_per_element) / (torch.sum(actual_values_per_element) + epsilon)).item() * 100
    smape_overall = (200.0 * torch.sum(smape_numerator_per_element) / (torch.sum(actual_values_per_element + torch.abs(pred_t)) + epsilon)).item()

    # FID 計算
    fid = float('nan')
    if all_generated_norm and all_target_norm:
        gen_fid_tensor = torch.cat(all_generated_norm, dim=0)[:max_fid_samples_actual]
        real_fid_tensor = torch.cat(all_target_norm, dim=0)[:max_fid_samples_actual]
        num_fid = min(gen_fid_tensor.shape[0], real_fid_tensor.shape[0])
        if num_fid > 1:
            logger.info(f"正在為 baseline_model 計算 FID (樣本數: {num_fid})...")
            act_gen = get_activations(gen_fid_tensor, inception_model_fid, config["device"], config["fid_batch_size"])
            act_real = get_activations(real_fid_tensor, inception_model_fid, config["device"], config["fid_batch_size"])
            fid = calculate_fid(act_real, act_gen)
    
    results = {
        "mse": mse, "mae": mae,
        "mape_avg_grid": mape_avg_grid, "smape_avg_grid": smape_avg_grid,
        "mape_overall": mape_overall, "smape_overall": smape_overall,
        "fid": fid
    }
    logger.info(f"Metrics for {config['model_name']} ({prefix}): {results}")

    # 計算逐網格誤差
    pred_squeezed = pred_t.squeeze(1).squeeze(1)
    target_squeezed = target_t.squeeze(1).squeeze(1)
    mse_g = torch.mean((pred_squeezed - target_squeezed)**2, dim=0).cpu().numpy()
    mae_g = torch.mean(torch.abs(pred_squeezed - target_squeezed), dim=0).cpu().numpy()
    mape_g_t = torch.abs((target_squeezed - pred_squeezed) / (torch.abs(target_squeezed) + epsilon)) * 100
    mape_g = torch.mean(mape_g_t, dim=0).cpu().numpy()
    smape_n_g = torch.abs(pred_squeezed - target_squeezed)
    smape_d_g = (torch.abs(target_squeezed) + torch.abs(pred_squeezed))/2.0 + epsilon
    smape_g_t = (smape_n_g / smape_d_g) * 100
    smape_g = torch.mean(smape_g_t, dim=0).cpu().numpy()
    error_grids = {'MSE': mse_g.flatten(), 'MAE': mae_g.flatten(), 'MAPE': mape_g.flatten(), 'SMAPE': smape_g.flatten()}

    # 視覺化
    visualize_predictions(pred_t, target_t, config, prefix=f"{prefix}_comparison")
    plot_grid_with_error(dataloader.dataset, error_grids, config, prefix=f"{prefix}_error_maps")
    
    return results, error_grids

def export_baseline_evaluation_to_excel(
    results: Dict[str, float],
    error_grids: Dict[str, np.ndarray],
    config: Dict[str, Any],
    prefix: str = "eval"
):
    """將 Baseline 模型的評估結果匯出為 Excel 檔案。"""
    save_dir = config["model_save_dir"]
    model_name = config["model_name"]
    logger.info(f"正在為 {model_name} ({prefix}) 準備 Excel 報告...")

    excel_rows_to_export = []
    
    # 從 CONFIG 中獲取繪圖所需的網格資訊
    num_grid_cells = config["H"] * config["W"]
    grid_idx_to_rc_map = config.get("cached_basemodel_grid_idx_to_rc_map")
    sorted_flow_columns = config.get("cached_basemodel_sorted_flow_columns")
    selected_sensor_info = config.get("cached_basemodel_selected_sensor_info")
    sensor_info_lookup = {info['name']: {'lon': info['lon'], 'lat': info['lat']} for info in selected_sensor_info} if selected_sensor_info else {}

    # --- 逐網格數據 ---
    excel_rows_to_export.append({'資料來源': f"--- {model_name} ({prefix}) 逐網格誤差 ---"})
    for flat_idx in range(num_grid_cells):
        grid_r, grid_c = grid_idx_to_rc_map.get(flat_idx, ('N/A', 'N/A')) if grid_idx_to_rc_map else ('N/A', 'N/A')
        col_name = sorted_flow_columns[flat_idx] if sorted_flow_columns and flat_idx < len(sorted_flow_columns) else 'N/A'
        lon, lat = (sensor_info_lookup.get(col_name, {}).get('lon'), sensor_info_lookup.get(col_name, {}).get('lat'))
        
        row_data = {
            '資料來源': model_name,
            '網格座標_R': grid_r, '網格座標_C': grid_c, '經度': lon, '緯度': lat,
            'MSE': error_grids.get('MSE')[flat_idx] if 'MSE' in error_grids and flat_idx < len(error_grids['MSE']) else np.nan,
            'MAE': error_grids.get('MAE')[flat_idx] if 'MAE' in error_grids and flat_idx < len(error_grids['MAE']) else np.nan,
            'MAPE': error_grids.get('MAPE')[flat_idx] if 'MAPE' in error_grids and flat_idx < len(error_grids['MAPE']) else np.nan,
            'SMAPE': error_grids.get('SMAPE')[flat_idx] if 'SMAPE' in error_grids and flat_idx < len(error_grids['SMAPE']) else np.nan,
        }
        excel_rows_to_export.append(row_data)

    # --- 整體平均指標 ---
    excel_rows_to_export.append({'資料來源': f"--- {model_name} ({prefix}) 整體指標 ---"})
    avg_row = {
        '資料來源': model_name, '網格座標_R': '整體平均',
        'MSE': results.get('mse', np.nan), 'MAE': results.get('mae', np.nan),
        'MAPE (AvgGrid)': results.get('mape_avg_grid', np.nan), 
        'SMAPE (AvgGrid)': results.get('smape_avg_grid', np.nan),
        'MAPE (Overall)': results.get('mape_overall', np.nan),
        'SMAPE (Overall)': results.get('smape_overall', np.nan),
        'FID': results.get('fid', np.nan)
    }
    excel_rows_to_export.append(avg_row)

    # --- 寫入 Excel ---
    df_export = pd.DataFrame(excel_rows_to_export)
    excel_column_order = ['資料來源', '網格座標_R', '網格座標_C', '經度', '緯度', 
                             'MSE', 'MAE', 'MAPE', 'SMAPE', 
                             'MAPE (AvgGrid)', 'SMAPE (AvgGrid)', 
                             'MAPE (Overall)', 'SMAPE (Overall)', 'FID']
    df_export = df_export.reindex(columns=excel_column_order) # 確保欄位順序正確
    
    excel_final_path = os.path.join(save_dir, f"{prefix}_metrics_detailed.xlsx")
    try:
        df_export.to_excel(excel_final_path, index=False, sheet_name=f"{model_name}_Details")
        logger.info(f"詳細評估指標已匯出至 Excel: {excel_final_path}")
    except Exception as e:
        logger.error(f"匯出 Excel 失敗: {e}")
#%%
# ==============================================================================
# 主程式執行流程
# ==============================================================================
if __name__ == '__main__':
    logger.info(f"===== DDPM Baseline 5-Channel Model - Training and Evaluation =====")
    
    # 為了日誌記錄，創建一個不包含敏感或過長資訊的 CONFIG 副本
    config_for_log = CONFIG.copy()
    # 移除可能過長的快取資訊
    keys_to_remove = ["cached_basemodel_sorted_flow_columns", "cached_basemodel_selected_sensor_info", "cached_basemodel_grid_idx_to_rc_map"]
    for key in keys_to_remove:
        config_for_log.pop(key, None)
    logger.info(f"Full CONFIG (selected fields): {json.dumps(config_for_log, indent=2, ensure_ascii=False)}")

    # --- 載入並預處理數據 ---
    full_df = pd.read_csv(CONFIG["data_path"])
    if 'hoilday' in full_df.columns and 'holiday' not in full_df.columns:
        full_df.rename(columns={'hoilday': 'holiday'}, inplace=True)
    logger.info(f"已載入資料: {CONFIG['data_path']}. 形狀: {full_df.shape}")

    # --- 創建組合特徵 ---
    if '月' in full_df.columns and '日' in full_df.columns:
        full_df['month_day_combined'] = full_df['月'] * 100 + full_df['日']
        logger.info("已成功創建 'month_day_combined' 組合特徵欄位。")
    if 'weekday' not in full_df.columns and '日期' in full_df.columns:
        try:
            full_df['日期'] = pd.to_datetime(full_df['日期'])
            full_df['weekday'] = full_df['日期'].dt.dayofweek
            logger.info("已成功創建 'weekday' 特徵欄位。")
        except Exception as e:
            logger.error(f"從 '日期' 創建 'weekday' 失敗: {e}")
    
    # --- 讀取 Basemodel 以獲取網格資訊 ---
    # Baseline 模型雖然不使用 Basemodel 的輸出，但需要其網格欄位資訊來建構目標
    logger.info("從 Basemodel 檢查點載入網格資訊...")
    basemodel_chkpt_path = CONFIG.get("basemodel_checkpoint")





    if not basemodel_chkpt_path or not os.path.exists(basemodel_chkpt_path):
        raise FileNotFoundError(f"未找到 Basemodel 檢查點: {basemodel_chkpt_path}。需要它來獲取網格欄位資訊。")
    chkpt_basemodel = torch.load(basemodel_chkpt_path, map_location=CONFIG["device"], weights_only=False)
    CONFIG["cached_basemodel_sorted_flow_columns"] = chkpt_basemodel.get('sorted_flow_columns')
    CONFIG["cached_basemodel_grid_idx_to_rc_map"] = chkpt_basemodel.get('grid_idx_to_rc_map')
    CONFIG["cached_basemodel_selected_sensor_info"] = chkpt_basemodel.get('selected_sensor_info')
    if not CONFIG["cached_basemodel_sorted_flow_columns"]:
        raise ValueError("Basemodel 檢查點缺少 'sorted_flow_columns'。")
    logger.info("成功從 Basemodel 檢查點加載網格欄位資訊。")


    # --- 根據 CONFIG 中的規則過濾數據 ---
    logger.info(f"===== Baseline Model: 根據規則過濾數據 =====")
    event_filter_config = CONFIG["stage4_config"]["event_filter"] # 從 stage4_config 獲取
    if event_filter_config["enabled"]:
        event_file_path = event_filter_config["file_path"]
        month_col = event_filter_config["month_col"]
        day_col = event_filter_config["day_col"]
        
        if not os.path.exists(event_file_path):
            raise FileNotFoundError(f"找不到活動日期 Excel 檔案: {event_file_path}")
        
        logger.info(f"正在從 {event_file_path} 讀取活動日期...")
        events_df = pd.read_excel(event_file_path)

        if not all(col in events_df.columns for col in [month_col, day_col]):
            raise ValueError(f"Excel 檔案 '{event_file_path}' 中缺少必要的欄位:'{month_col}', 或 '{day_col}'")

        event_month_day_set = set(zip(events_df[month_col], events_df[day_col]))
        logger.info(f"從檔案中提取了 {len(event_month_day_set)} 個不重複的活動日期 (月, 日)。")

        final_mask = full_df.apply(lambda row: (row['月'], row['日']) in event_month_day_set, axis=1)
    else:
        final_mask = pd.Series(True, index=full_df.index)

    df_for_baseline = full_df[final_mask].copy()
    logger.info(f"數據過濾完成。將使用 {len(df_for_baseline)} 筆資料進行訓練和評估。")
    
    # --- 數據準備與 DataLoader 創建 ---
    logger.info(f"===== Baseline Model: 數據集準備 =====")
    indices_all = np.arange(len(df_for_baseline))
    np.random.shuffle(indices_all)
    train_len = int(CONFIG["train_split_ratio"] * len(indices_all))
    val_len = int(CONFIG["val_split_ratio"] * len(indices_all))
    train_indices = indices_all[:train_len]
    val_indices = indices_all[train_len : train_len + val_len]
    test_indices = indices_all[train_len + val_len:]
    
    train_dataset = BaselineDataset(
        df_for_processing=df_for_baseline.iloc[train_indices],
        config=CONFIG,
        mode='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    logger.info(f"Baseline 訓練數據集創建，含 {len(train_dataset)} 樣本。")

    val_loader = None
    if len(val_indices) > 0:
        val_dataset = BaselineDataset(
            df_for_processing=df_for_baseline.iloc[val_indices],
            config=CONFIG, mode='val',
            norm_stats_from_train=train_dataset.norm_stats_dict,
            target_info_from_train={"avg_flow_map": train_dataset.average_flow_map_dict, "norm_stats": train_dataset.norm_stats_target}
        )
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False,
                                num_workers=CONFIG["num_workers"], pin_memory=True)
        logger.info(f"Baseline 驗證數據集創建，含 {len(val_dataset)} 樣本。")

    test_loader = None
    if len(test_indices) > 0:
        test_dataset = BaselineDataset(
            df_for_processing=df_for_baseline.iloc[test_indices],
            config=CONFIG, mode='test',
            norm_stats_from_train=train_dataset.norm_stats_dict,
            target_info_from_train={"avg_flow_map": train_dataset.average_flow_map_dict, "norm_stats": train_dataset.norm_stats_target}
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False,
                                 num_workers=CONFIG["num_workers"], pin_memory=True)
        logger.info(f"Baseline 測試數據集創建，含 {len(test_dataset)} 樣本。")

    # --- 模型初始化 ---
    logger.info(f"===== Baseline Model: 模型初始化 ({CONFIG['model_name']}) =====")
    baseline_unet = UNet3D(
        input_image_channels=CONFIG["image_channels"], base_channels=CONFIG["base_channels_unet"],
        time_emb_dim=CONFIG["time_emb_dim"], condition_encode_dim=CONFIG["condition_encode_dim"],
        dropout_rate=CONFIG["unet_dropout_rate"]
    ).to(CONFIG["device"])

    baseline_model = DDPM3D(
        unet_model=baseline_unet, timesteps=CONFIG["timesteps"],
        image_size=(CONFIG["D"], CONFIG["H"], CONFIG["W"]),
        image_channels=CONFIG["image_channels"],
        condition_input_channels=CONFIG["condition_input_channels"],
        condition_encode_dim=CONFIG["condition_encode_dim"],
        device=CONFIG["device"]
    )
   #%% 
    # --- 訓練迴圈 ---
    if train_loader and len(train_loader.dataset) > 0:
        logger.info(f"===== Baseline Model: 模型訓練開始 =====")
        optimizer = optim.AdamW(baseline_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG.get("weight_decay", 0.0))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CONFIG["lr_scheduler_factor"], 
                                      patience=CONFIG["lr_scheduler_patience"], min_lr=CONFIG["lr_scheduler_min_lr"])
        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        start_epoch = 1
        
        model_checkpoint_path = CONFIG["checkpoint_full_path"]

        if os.path.exists(model_checkpoint_path):
            logger.info(f"從檢查點恢復訓練: {model_checkpoint_path}")
            checkpoint = torch.load(model_checkpoint_path, map_location=CONFIG["device"], weights_only=False)
            baseline_model.load_state_dict(checkpoint['ddpm_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"將從 Epoch {start_epoch} 繼續訓練。")

        for epoch in range(start_epoch, CONFIG["epochs"] + 1):
            baseline_model.train()
            total_train_loss = 0.0
            pbar_train = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

            for target_b, condition_b in pbar_train:
                target_b = target_b.to(CONFIG["device"])
                condition_b = condition_b.to(CONFIG["device"])

                optimizer.zero_grad()
                t_b = torch.randint(0, baseline_model.timesteps, (target_b.shape[0],), device=CONFIG["device"]).long()
                loss = baseline_model.p_losses(target_b, t_b, condition_b)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar_train.set_postfix({"Batch Loss": f"{loss.item():.5f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)

            avg_val_loss = float('inf')

            baseline_model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                # 遍歷驗證數據集
                for target_v, condition_v in val_loader:
                    target_v = target_v.to(CONFIG["device"])
                    condition_v = condition_v.to(CONFIG["device"])

                    # 為驗證批次隨機選擇一個時間步 t
                    t_v = torch.randint(0, baseline_model.timesteps, (target_v.shape[0],), device=CONFIG["device"]).long()
                    
                    # 使用與訓練時相同的 p_losses 函式計算損失，快速且穩定
                    val_loss = baseline_model.p_losses(target_v, t_v, condition_v)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                torch.save({
                    'epoch': epoch,
                    'ddpm_state_dict': baseline_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config_snapshot_at_save': CONFIG,
                    'cond_norm_stats': train_dataset.norm_stats_dict,
                    'target_norm_stats': train_dataset.norm_stats_target,
                    'target_avg_flow_map': train_dataset.average_flow_map_dict
                }, model_checkpoint_path)
                tqdm.write(f"Epoch {epoch}: 新最佳 Baseline 模型已儲存 (Val Loss: {best_val_loss:.5f})。")
            else:
                early_stopping_counter += 1
        
            tqdm.write(f"Epoch {epoch}/{CONFIG['epochs']} - Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}, LR: {optimizer.param_groups[0]['lr']:.1e}, ES: {early_stopping_counter}/{CONFIG['early_stopping_patience']}")

            if early_stopping_counter >= CONFIG['early_stopping_patience']:
                logger.info(f"訓練因早停機制觸發於 Epoch {epoch}。")
                break
    else:
        logger.info("跳過 Baseline 模型訓練，訓練數據為空。")
#%%
    # --- 最終評估 ---
    logger.info(f"===== Baseline Model: 最終評估 =====")
    if test_loader and len(test_loader.dataset) > 0:
        best_model_path = CONFIG["checkpoint_full_path"]
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"未找到用於評估的最佳模型檢查點: {best_model_path}")
        
        logger.info(f"載入最佳 Baseline 模型: {best_model_path}")
        chkpt = torch.load(best_model_path, map_location=CONFIG["device"], weights_only=False)
        baseline_model.load_state_dict(chkpt['ddpm_state_dict'])

        inception_model_for_fid = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True).to(CONFIG["device"])
        inception_model_for_fid.fc = nn.Identity()
        inception_model_for_fid.eval()
        
        final_metrics, final_error_grids = evaluate_baseline_model(
            model_trained=baseline_model, dataloader=test_loader,
            inception_model_fid=inception_model_for_fid, config=CONFIG,
            target_norm_stats=train_dataset.norm_stats_target,
            max_samples_for_fid=CONFIG.get("fid_num_samples"),
            prefix=f"final_baseline_evaluation"
        )
        
        # --- [新增] 呼叫 Excel 匯出函式 ---
        export_baseline_evaluation_to_excel(
            results=final_metrics,
            error_grids=final_error_grids,
            config=CONFIG,
            prefix="final_baseline_evaluation"
        )
        
        metrics_save_path = os.path.join(CONFIG["model_save_dir"], f"final_baseline_evaluation_metrics.json")
        with open(metrics_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=4, ensure_ascii=False)
        logger.info(f"Baseline 模型最終評估指標 (JSON) 已儲存到: {metrics_save_path}")
    else:
        logger.warning("由於缺少測試數據，跳過最終評估。")

    logger.info("===== Baseline 模型流程結束 =====")
# %%

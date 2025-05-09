# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import re
import math
import json
import logging
import random
import numpy as np
import pandas as pd
import scipy.linalg # For FID
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
from scipy.optimize import linear_sum_assignment # For Hungarian Algorithm
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm # Progress bars

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # --- Data Parameters ---
    "data_path": "all_merged_sample.csv", # 您的資料路徑
    "H": 20, # 網格高度
    "W": 20, # 網格寬度
    "D": 1,  # 網格深度 (流量圖本身為1)

    # --- Model Parameters ---
    "image_channels": 1,      # 主要數據(流量圖)的通道數
    "condition_input_channels": 2, # 條件處理器接收的原始條件通道數 (小時網格 + 星期幾網格)
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 (可調超參數)
    "base_channels_unet": 64,   # UNet3D 的基礎通道數
    "time_emb_dim": 256,        # 時間嵌入的維度

    # --- DDPM Parameters ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- Training Parameters ---
    "epochs": 200, # 可根據需要調整
    "batch_size": 16, # 可根據 GPU 記憶體調整
    "lr": 1e-4, # 學習率
    "num_workers": 0, # DataLoader 的工作執行緒 (Windows建議0, Linux可>0)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42, # 隨機種子

    # --- Evaluation Parameters ---
    "eval_batch_size": 16,
    "fid_batch_size": 32,
    "fid_num_samples": 500, # 用於 FID 計算的樣本數

    # --- Paths and Saving ---
    "save_dir": "results_ddpm_conditioned_flow_taipei_extra_v2", # 更新儲存目錄
    "plot_grid_mapping_path": "grid_mapping_visualization_taipei.png",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
logger.info(f"結果將儲存於: {CONFIG['save_dir']}")

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])
logger.info(f"使用裝置: {CONFIG['device']}")

# ==============================================================================
# DATASET CLASS (PeopleFlowDatasetCondition) - 整合 extra_data 處理 (extra_data不標準化)
# ==============================================================================
class PeopleFlowDatasetCondition(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 config: Dict[str, Any],
                 mode: str = 'train',
                 # For val/test, these are passed from training_dataset instance
                 average_flow_map_dict: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                 norm_stats_flow: Optional[Dict[str, float]] = None, # FOR FLOW DATA STANDARDIZATION
                 sorted_flow_columns: Optional[List[str]] = None,
                 grid_idx_to_rc_map: Optional[Dict[int, Tuple[int,int]]] = None,
                 # extra_cont_mean, extra_cont_std are NO LONGER USED as extra_data is not standardized
                 processed_extra_columns: Optional[List[str]] = None # Names of processed extra_data columns
                ):
        super().__init__()
        self.df_original = df.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.H = config["H"]
        self.W = config["W"]
        self.D = config["D"]
        self.image_channels = config["image_channels"]
        self.num_grid_cells = self.H * self.W

        # --- Time parsing (from original DATATIME) ---
        if 'DATATIME' not in self.df_original.columns:
            raise ValueError("資料中未找到 'DATATIME' 欄位。")
        try:
            df_datetime_processed = self.df_original.copy()
            df_datetime_processed['DATATIME'] = pd.to_datetime(df_datetime_processed['DATATIME'])
            self.hours_original_np = df_datetime_processed['DATATIME'].dt.hour.values
            self.day_of_week_original_np = df_datetime_processed['DATATIME'].dt.dayofweek.values
        except Exception as e:
            raise ValueError(f"無法解析 'DATATIME' 欄位: {e}。")

        # --- Process extra_data (meteorological, holiday, time features etc.) ---
        df_for_extra_processing = self.df_original.copy()

        temp_dt_series = pd.to_datetime(df_for_extra_processing['DATATIME'])
        df_for_extra_processing['年'] = temp_dt_series.dt.year
        df_for_extra_processing['月'] = temp_dt_series.dt.month
        df_for_extra_processing['日'] = temp_dt_series.dt.day
        df_for_extra_processing['時'] = temp_dt_series.dt.hour # Raw hour
        df_for_extra_processing['weekday'] = temp_dt_series.dt.dayofweek # Raw weekday

        self.extra_cols_list_definition = [
            "測站氣壓", "海平面氣壓", "氣溫", "露點溫度", "相對溼度", "風速", "最大陣風",
            "降水量", "降水時數", "日照時數", "全天空日射量", "能見度", "紫外線指數", "總雲量",
            "holiday", "weekday", "年", "月", "日", "時"
        ]
        wind_cols_to_process = []
        if '風向' in df_for_extra_processing.columns: wind_cols_to_process.append('風向')
        if '最大陣風風向' in df_for_extra_processing.columns: wind_cols_to_process.append('最大陣風風向')

        for col in wind_cols_to_process:
            df_for_extra_processing[f'sin_{col}'] = np.sin(np.deg2rad(df_for_extra_processing[col]))
            df_for_extra_processing[f'cos_{col}'] = np.cos(np.deg2rad(df_for_extra_processing[col]))
            if f'sin_{col}' not in self.extra_cols_list_definition: self.extra_cols_list_definition.append(f'sin_{col}')
            if f'cos_{col}' not in self.extra_cols_list_definition: self.extra_cols_list_definition.append(f'cos_{col}')
        
        self.extra_cols_list_definition = [col for col in self.extra_cols_list_definition if col not in ['風向', '最大陣風風向']]
        
        df_extra_subset = df_for_extra_processing[present_extra_cols].copy()



        if "hoilday" in df_extra_subset.columns:
            df_extra_subset.rename(columns={"hoilday": "holiday"}, inplace=True)
        
        cat_features = ['holiday']
        actual_cat_features = [col for col in cat_features if col in df_extra_subset.columns]
        
        if actual_cat_features:
            df_extra_subset[actual_cat_features] = df_extra_subset[actual_cat_features].astype(str)
            df_cat = pd.get_dummies(df_extra_subset[actual_cat_features], prefix=actual_cat_features, dummy_na=False)
        else:
            df_cat = pd.DataFrame(index=df_extra_subset.index)

        df_cont = df_extra_subset.drop(columns=actual_cat_features, errors='ignore')
        
        # NO STANDARDIZATION for continuous extra_data features
        # df_cont_processed = df_cont
        
        df_extra_processed = pd.concat([df_cont, df_cat], axis=1) # Use raw continuous + one-hot categorical
        
        if self.mode == 'train':
            self.processed_extra_columns = list(df_extra_processed.columns)
            self.processed_extra_data_np = df_extra_processed.fillna(0).values.astype(np.float32)
            logger.info(f"訓練集: 已處理 {len(self.processed_extra_columns)} 個額外特徵 (連續特徵未標準化)。")
        else:
            if processed_extra_columns is None:
                raise ValueError("對於 val/test 模式，必須提供 processed_extra_columns。")
            self.processed_extra_columns = processed_extra_columns
            # Ensure columns are in the same order and fill missing ones (e.g. from one-hot)
            df_extra_processed = df_extra_processed.reindex(columns=self.processed_extra_columns, fill_value=0)
            self.processed_extra_data_np = df_extra_processed.fillna(0).values.astype(np.float32)
            logger.info(f"{self.mode} 資料集: 已處理額外特徵 (連續特徵未標準化)。")

        # --- Flow data grid mapping and average calculation ---
        if self.mode == 'train':
            all_available_sensor_info = self._extract_all_sensor_info_from_csv()
            self.selected_sensor_info, selected_real_coords_np = self._select_sensors(all_available_sensor_info)
            self.grid_target_coords, self.grid_idx_to_rc_map = self._define_target_grid_cells_hierarchical_style(selected_real_coords_np)
            self.sorted_flow_columns = self._map_sensors_to_target_grid_hungarian(
                self.selected_sensor_info, selected_real_coords_np, self.grid_target_coords
            )
            self._plot_grid_mapping(
                selected_real_coords_np, self.grid_target_coords, self.grid_idx_to_rc_map, self.sorted_flow_columns,
                os.path.join(self.config["save_dir"], self.config["plot_grid_mapping_path"])
            )
            self.average_flow_map_dict = self._calculate_average_flows()

            all_avg_flows_list = [flow for flow in self.average_flow_map_dict.values() if flow is not None]
            if not all_avg_flows_list:
                raise ValueError("訓練集中未計算出任何平均流量。無法計算流量標準化統計量。")
            all_avg_flows_np = np.stack(all_avg_flows_list)
            self.flow_mean_val = np.mean(all_avg_flows_np) # FOR FLOW DATA STANDARDIZATION
            self.flow_std_val = np.std(all_avg_flows_np)   # FOR FLOW DATA STANDARDIZATION
            if self.flow_std_val < 1e-5: self.flow_std_val = 1e-5
            self.norm_stats_flow = {'mean': self.flow_mean_val, 'std': self.flow_std_val}
            logger.info(f"訓練集流量標準化統計量: 平均值={self.flow_mean_val:.4f}, 標準差={self.flow_std_val:.4f}")

        else: 
            if not all([average_flow_map_dict, norm_stats_flow, sorted_flow_columns, grid_idx_to_rc_map]):
                raise ValueError("average_flow_map_dict, norm_stats_flow, sorted_flow_columns, grid_idx_to_rc_map 必須為 val/test 模式提供。")
            self.average_flow_map_dict = average_flow_map_dict
            self.norm_stats_flow = norm_stats_flow # Use passed flow norm stats
            self.flow_mean_val = self.norm_stats_flow['mean']
            self.flow_std_val = self.norm_stats_flow['std']
            self.sorted_flow_columns = sorted_flow_columns
            self.grid_idx_to_rc_map = grid_idx_to_rc_map
            logger.info(f"使用預計算的流量標準化統計量: 平均值={self.flow_mean_val:.4f}, 標準差={self.flow_std_val:.4f}")

    # Helper methods _extract_all_sensor_info_from_csv, _select_sensors, 
    # _define_target_grid_cells_hierarchical_style, _map_sensors_to_target_grid_hungarian,
    # _calculate_average_flows, _plot_grid_mapping remain THE SAME as the previous version I provided.
    # I will paste them here again for completeness of this class.

    def _extract_all_sensor_info_from_csv(self) -> List[Dict[str, Any]]:
        all_sensor_info = []
        max_sensor_idx = -1
        for col in self.df_original.columns:
            if col.startswith('flow_') or col.startswith('latitude_') or col.startswith('longitude_'):
                try: idx = int(col.split('_')[-1]); max_sensor_idx = max(max_sensor_idx, idx)
                except ValueError: continue
        if max_sensor_idx == -1: raise ValueError("無法從CSV欄位名確定感測器索引。")
        # logger.info(f"在CSV中找到最大感測器索引為 {max_sensor_idx}。") # Logged once is enough
        for i in range(max_sensor_idx + 1):
            fcol, latcol, loncol = f'flow_{i}', f'latitude_{i}', f'longitude_{i}'
            if all(c in self.df_original.columns for c in [fcol, latcol, loncol]):
                lat, lon = self.df_original[latcol].iloc[0], self.df_original[loncol].iloc[0]
                if pd.notna(lat) and pd.notna(lon):
                    all_sensor_info.append({'name': fcol, 'lon': float(lon), 'lat': float(lat), 'original_csv_sensor_index': i})
        if not all_sensor_info: raise ValueError("無法從CSV提取有效感測器數據。")
        # logger.info(f"提取到 {len(all_sensor_info)} 個感測器資訊。")
        return all_sensor_info

    def _select_sensors(self, all_sensor_info: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        num_required = self.num_grid_cells
        if len(all_sensor_info) < num_required: raise ValueError(f"網格點數({num_required}) > 可用感測器({len(all_sensor_info)})")
        coords = np.array([(s['lon'], s['lat']) for s in all_sensor_info])
        if len(all_sensor_info) > num_required:
            logger.info(f"可用感測器({len(all_sensor_info)}) > 所需({num_required})。選擇最近中心的點。")
            center = np.mean(coords, axis=0)
            dists = np.sum((coords - center)**2, axis=1)
            sel_indices = np.argsort(dists)[:num_required]
            sel_info = [all_sensor_info[i] for i in sel_indices]
            sel_coords = coords[sel_indices]
        else:
            sel_info = all_sensor_info; sel_coords = coords
        logger.info(f"已選定 {len(sel_info)} 個感測器進行網格映射。")
        return sel_info, sel_coords

    def _define_target_grid_cells_hierarchical_style(self, selected_real_coords_np: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
        if selected_real_coords_np.shape[0] != self.num_grid_cells:
            raise ValueError(f"selected_real_coords_np 應含 {self.num_grid_cells} 點, 得到 {selected_real_coords_np.shape[0]}。")
        logger.info("使用 'hierarchical' 風格定義目標網格中心點...")
        center_lon, center_lat = np.mean(selected_real_coords_np, axis=0)
        unique_lons, unique_lats = np.unique(selected_real_coords_np[:,0]), np.unique(selected_real_coords_np[:,1])
        lon_diffs, lat_diffs = np.diff(np.sort(unique_lons)), np.diff(np.sort(unique_lats))
        lon_step = np.median(lon_diffs[lon_diffs > 1e-6]) if len(lon_diffs[lon_diffs > 1e-6]) > 0 else 0.001
        lat_step = np.median(lat_diffs[lat_diffs > 1e-6]) if len(lat_diffs[lat_diffs > 1e-6]) > 0 else 0.001
        if lon_step <= 1e-6: lon_step = 0.001
        if lat_step <= 1e-6: lat_step = 0.001
        logger.info(f"目標網格中心: (lon:{center_lon:.4f}, lat:{center_lat:.4f}), 步長: (lon:{lon_step:.6f}, lat:{lat_step:.6f})")
        
        grid_targets = np.zeros((self.num_grid_cells, 2))
        idx_to_rc = {}
        idx = 0
        for r_idx in range(self.H):
            for c_idx in range(self.W):
                tlon = center_lon + (c_idx - (self.W-1)/2.0) * lon_step
                tlat = center_lat - (r_idx - (self.H-1)/2.0) * lat_step
                grid_targets[idx, 0], grid_targets[idx, 1] = tlon, tlat
                idx_to_rc[idx] = (r_idx,c_idx)
                idx += 1
        return grid_targets, idx_to_rc

    def _map_sensors_to_target_grid_hungarian(self, sel_info: List[Dict[str,Any]], sel_coords: np.ndarray, grid_targets: np.ndarray) -> List[str]:
        logger.info("使用匈牙利演算法將選定感測器映射到目標網格...")
        n = self.num_grid_cells
        if not (sel_coords.shape[0] == n and grid_targets.shape[0] == n and len(sel_info) == n):
            raise ValueError("匈牙利分配時，輸入數量必須都等於 H*W。")
        costs = np.sum((sel_coords[:, np.newaxis, :] - grid_targets[np.newaxis, :, :])**2, axis=2)
        costs = np.sqrt(costs)
        real_indices, target_indices = linear_sum_assignment(costs)
        
        target_to_real_map = {t_idx: r_idx for r_idx, t_idx in zip(real_indices, target_indices)}
        sorted_cols = [""] * n
        for flat_target_idx in range(n):
            if flat_target_idx in target_to_real_map:
                real_idx_in_sel_list = target_to_real_map[flat_target_idx]
                sorted_cols[flat_target_idx] = sel_info[real_idx_in_sel_list]['name']
            else: raise Exception(f"目標網格 {flat_target_idx} 未分配到感測器。")
        logger.info(f"成功為網格排序 {len(sorted_cols)} 個流量欄位。")
        return sorted_cols

    def _calculate_average_flows(self) -> Dict[Tuple[int, int], np.ndarray]:
        logger.info("計算 (小時, 星期幾) 平均流量圖...")
        avg_flows = {}
        for col in self.sorted_flow_columns:
            if col not in self.df_original.columns: raise ValueError(f"流量欄位 '{col}' 在 DataFrame 未找到。")
        
        flow_data_grid_alltimes = self.df_original[self.sorted_flow_columns].values.astype(np.float32)
        
        grouping_df = pd.DataFrame({'hour': self.hours_original_np, 'day_of_week': self.day_of_week_original_np})
        
        for (hr, dow), group_indices in grouping_df.groupby(['hour', 'day_of_week']).groups.items():
            group_flows = flow_data_grid_alltimes[group_indices]
            mean_flow_flat = np.nanmean(group_flows, axis=0)
            mean_flow_flat[np.isnan(mean_flow_flat)] = 0
            avg_flows[(hr, dow)] = mean_flow_flat.reshape(self.H, self.W)
        if not avg_flows: logger.warning("未計算任何平均流量。")
        logger.info(f"計算完成 {len(avg_flows)} 個條件的平均流量。")
        return avg_flows

    def _plot_grid_mapping(self, sel_coords, grid_targets, idx_to_rc, sorted_cols, save_path):
        try:
            plt.figure(figsize=(10,10)); plt.style.use('seaborn_v0_8_whitegrid') # Or any other preferred style
            plt.scatter(sel_coords[:,0], sel_coords[:,1], c='blue', marker='o', s=25, alpha=0.7, label='Selected Actual Sensor Locations', zorder=2)
            plt.scatter(grid_targets[:,0], grid_targets[:,1], c='red', marker='x', s=25, alpha=0.7, label='Target Grid Cell Centers', zorder=3)
            for flat_idx in range(self.num_grid_cells):
                r_idx, c_idx = idx_to_rc.get(flat_idx, (-1,-1))
                if r_idx != -1: plt.text(grid_targets[flat_idx,0], grid_targets[flat_idx,1], f'T[{r_idx},{c_idx}]', fontsize=5,color='darkred',ha='center',va='bottom',zorder=4)
            plt.xlabel("Longitude (Taipei City)"); plt.ylabel("Latitude"); plt.title(f"Grid Mapping ({self.H}x{self.W})")
            plt.legend(); plt.savefig(save_path, dpi=200); plt.close()
            logger.info(f"網格映射視覺化圖儲存至 {save_path}")
        except Exception as e: logger.error(f"繪製網格圖出錯: {e}", exc_info=True)

    def __len__(self) -> int:
        return len(self.df_original)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        # --- Target Average Flow (x_start for DDPM) - STANDARDIZED ---
        current_hour_original = self.hours_original_np[idx]
        current_dow_original = self.day_of_week_original_np[idx]

        target_avg_flow_np = self.average_flow_map_dict.get((current_hour_original, current_dow_original))
        if target_avg_flow_np is None:
            target_avg_flow_np = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Standardize the flow data
        standardized_avg_flow_np = (target_avg_flow_np - self.flow_mean_val) / self.flow_std_val
        target_flow_tensor = torch.from_numpy(standardized_avg_flow_np).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

        # --- Conditional Inputs: Original Hour (0-23) and Day of Week (0-6) ---
        # These will be converted to normalized grids inside DDPM3D
        
        # --- Processed Extra Data row (NOT STANDARDIZED for continuous features) ---
        extra_data_row_tensor = torch.from_numpy(self.processed_extra_data_np[idx]).float() # (num_processed_extra_features,)

        return target_flow_tensor, int(current_hour_original), int(current_dow_original), extra_data_row_tensor

# ==============================================================================
# UNet3D, DDPM3D, FID, Evaluation, and Main training loop
# (The UNet3D, SinusoidalTimeEmbedding, DoubleConv3D, Down3D, Up3D, OutConv3D
#  linear_beta_schedule, FID functions, evaluate_model function structure
#  will be the same as the full script I provided in the previous response when you asked
#  me to start coding "DDPM_Long-term.py".
#  The key changes are in DDPM3D's methods that receive hour/day scalars,
#  and the main training/evaluation loops calling these methods.)
# ==============================================================================

# Placeholder for UNet3D building blocks and UNet3D class
# (Assume they are defined as in the previous complete script output)
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
    def __init__(self, in_channels: int, out_channels: int): super().__init__(); self.maxpool_conv = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), DoubleConv3D(in_channels, out_channels))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.maxpool_conv(x)

class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__(); self.bilinear = bilinear
        if bilinear: self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True); self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else: self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2)); self.conv = DoubleConv3D(in_channels, out_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1); diffY = x2.size()[3] - x1.size()[3]; diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, 0, 0]); x = torch.cat([x2, x1], dim=1); return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int): super().__init__(); self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, input_image_channels: int, base_channels: int = 64, time_emb_dim: int = 256, condition_encode_dim: Optional[int] = None, bilinear_upsample: bool = True):
        super().__init__(); self.input_image_channels = input_image_channels; self.condition_encode_dim = condition_encode_dim or 0
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        actual_in_channels = self.input_image_channels + self.condition_encode_dim
        self.inc = DoubleConv3D(actual_in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels*2); self.down2 = Down3D(base_channels*2, base_channels*4); self.down3 = Down3D(base_channels*4, base_channels*8)
        factor = 2 if bilinear_upsample else 1; self.down4 = Down3D(base_channels*8, base_channels*16 // factor)
        self.up1 = Up3D(base_channels*16, base_channels*8 // factor, bilinear_upsample); self.up2 = Up3D(base_channels*8, base_channels*4 // factor, bilinear_upsample)
        self.up3 = Up3D(base_channels*4, base_channels*2 // factor, bilinear_upsample); self.up4 = Up3D(base_channels*2, base_channels, bilinear_upsample)
        self.outc = OutConv3D(base_channels, self.input_image_channels)
    def _add_time_embedding(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor: t_emb_expanded = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1); return x + t_emb_expanded
    def forward(self, x_t: torch.Tensor, time_steps: torch.Tensor, processed_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(time_steps)
        if processed_condition is not None:
            if x_t.shape[2:] != processed_condition.shape[2:]: raise ValueError(f"x_t DHW {x_t.shape[2:]} != processed_condition DHW {processed_condition.shape[2:]}")
            x_input = torch.cat((x_t, processed_condition), dim=1)
        else: x_input = x_t # This branch assumes condition_encode_dim was 0
        x1 = self.inc(x_input); x1 = self._add_time_embedding(x1, t_emb)
        x2 = self.down1(x1); x2 = self._add_time_embedding(x2, t_emb)
        x3 = self.down2(x2); x3 = self._add_time_embedding(x3, t_emb)
        x4 = self.down3(x3); x4 = self._add_time_embedding(x4, t_emb)
        x5 = self.down4(x4); x5 = self._add_time_embedding(x5, t_emb)
        x = self.up1(x5, x4); x = self._add_time_embedding(x, t_emb)
        x = self.up2(x, x3); x = self._add_time_embedding(x, t_emb)
        x = self.up3(x, x2); x = self._add_time_embedding(x, t_emb)
        x = self.up4(x, x1); x = self._add_time_embedding(x, t_emb)
        return self.outc(x)

def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM3D(nn.Module):
    def __init__(self,
                 unet_model: UNet3D,
                 timesteps: int,
                 image_size: Tuple[int, int, int], # (D, H, W)
                 image_channels: int,
                 condition_input_channels: int, # Raw channels for condition_processor (e.g., 2 for hour+day grids)
                 condition_encode_dim: int,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device: str = "cpu"):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps
        self.image_size_D, self.image_size_H, self.image_size_W = image_size # Store D, H, W
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
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()
        ).to(device)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]; out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        sact = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        soma_ct = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sact * x_start + soma_ct * noise

    def _prepare_conditional_input_grids(self,
                                        hour_scalars_batch: torch.Tensor, # (N,) original 0-23
                                        day_scalars_batch: torch.Tensor,  # (N,) original 0-6
                                        ) -> torch.Tensor: # Output (N, 2, D, H, W)
        batch_size = hour_scalars_batch.shape[0]
        norm_hours = hour_scalars_batch.float().to(self.device) / 23.0 # Normalize here
        norm_days = day_scalars_batch.float().to(self.device) / 6.0   # Normalize here

        hour_grids_list = [torch.full((self.image_size_H, self.image_size_W), norm_hours[i].item(), device=self.device, dtype=torch.float32) for i in range(batch_size)]
        day_grids_list = [torch.full((self.image_size_H, self.image_size_W), norm_days[i].item(), device=self.device, dtype=torch.float32) for i in range(batch_size)]
        
        hour_grids_t = torch.stack(hour_grids_list, dim=0).unsqueeze(1).unsqueeze(2) # (N,1,1,H,W)
        day_grids_t = torch.stack(day_grids_list, dim=0).unsqueeze(1).unsqueeze(2)   # (N,1,1,H,W)
        
        # Ensure depth matches self.image_size_D (which is 1)
        if self.image_size_D != 1: # Should not happen for this project
             hour_grids_t = hour_grids_t.repeat(1,1,self.image_size_D,1,1)
             day_grids_t = day_grids_t.repeat(1,1,self.image_size_D,1,1)
             
        return torch.cat((hour_grids_t, day_grids_t), dim=1) # (N, 2, D, H, W)

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, 
                 hour_scalars_batch: torch.Tensor, day_scalars_batch: torch.Tensor,
                 # extra_data_batch: torch.Tensor, # Not used directly by condition_processor currently
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        stacked_cond_grids = self._prepare_conditional_input_grids(hour_scalars_batch, day_scalars_batch)
        processed_condition = self.condition_processor(stacked_cond_grids)
        
        predicted_noise = self.model(x_t, t, processed_condition)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_scalar: int, t_tensor_batch: torch.Tensor, 
                 processed_conditions_batch: torch.Tensor) -> torch.Tensor:
        betas_t = self._extract(self.betas, t_tensor_batch, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor_batch, x_t.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t_tensor_batch, x_t.shape)
        
        predicted_noise = self.model(x_t, t_tensor_batch, processed_conditions_batch)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_scalar == 0: return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_tensor_batch, x_t.shape)
            noise = torch.randn_like(x_t); return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int,...], hour_scalars_batch: torch.Tensor, day_scalars_batch: torch.Tensor) -> torch.Tensor:
        batch_size = shape[0]; img = torch.randn(shape, device=self.device)
        stacked_cond_grids = self._prepare_conditional_input_grids(hour_scalars_batch, day_scalars_batch)
        processed_conditions = self.condition_processor(stacked_cond_grids)
        for i in tqdm(reversed(range(0, self.timesteps)), desc="DDPM Sampling Loop", total=self.timesteps, leave=False):
            t_tensor_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, i, t_tensor_batch, processed_conditions)
        return img

    @torch.no_grad()
    def sample(self, batch_size: int, hour_scalars_batch: torch.Tensor, day_scalars_batch: torch.Tensor) -> torch.Tensor:
        s = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        return self.p_sample_loop(s, hour_scalars_batch, day_scalars_batch)

# FID Functions (get_activations, calculate_frechet_distance, calculate_fid)
# These are assumed to be defined as in the previous complete script.
# ... (Paste FID functions here) ...
def get_activations(images: torch.Tensor, model: nn.Module, device: str, batch_size_fid: int = 32) -> np.ndarray:
    model.eval(); activations = []
    if images.shape[2] == 1: images_2d = images.squeeze(2)
    else: images_2d = images[:, :, images.shape[2]//2, :, :]; logger.warning("Image D > 1, taking middle slice for FID.")
    if images_2d.shape[1] == 1: images_2d = images_2d.repeat(1, 3, 1, 1)
    elif images_2d.shape[1] != 3 : images_2d = images_2d[:,:3,:,:]; logger.warning("Image C != 1 or 3, taking first 3 for FID.")
    
    transform_inception = transforms.Compose([transforms.Resize((299,299), antialias=True)])
    num_batches = math.ceil(images_2d.shape[0]/batch_size_fid)
    for i in range(num_batches):
        batch = transform_inception(images_2d[i*batch_size_fid:(i+1)*batch_size_fid].to(device))
        with torch.no_grad(): pred = model(batch)
        if isinstance(pred, tuple): pred = pred[0]
        activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)

def calculate_frechet_distance(mu1:np.ndarray, sigma1:np.ndarray, mu2:np.ndarray, sigma2:np.ndarray, eps:float=1e-6) -> float:
    mu1,mu2,sigma1,sigma2 = np.atleast_1d(mu1),np.atleast_1d(mu2),np.atleast_2d(sigma1),np.atleast_2d(sigma2)
    assert mu1.shape==mu2.shape and sigma1.shape==sigma2.shape
    diff=mu1-mu2; covmean_sqrt,_ = scipy.linalg.sqrtm(sigma1.dot(sigma2),disp=False)
    if not np.isfinite(covmean_sqrt).all():
        offset=np.eye(sigma1.shape[0])*eps; covmean_sqrt=scipy.linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    if np.iscomplexobj(covmean_sqrt): covmean_sqrt=covmean_sqrt.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean_sqrt)

def calculate_fid(real_acts:np.ndarray, gen_acts:np.ndarray)->float:
    mu_real, sigma_real = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu_gen, sigma_gen = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


# evaluate_model function
# (Assumed to be defined as in the previous complete script,
# but needs to pass scalar hour/day to ddpm_model.sample)
@torch.no_grad()
def evaluate_model(ddpm_model: DDPM3D, 
                   dataloader: DataLoader, 
                   inception_model_fid: nn.Module,
                   config: Dict[str, Any], 
                   max_samples_for_fid: Optional[int] = None
                   ) -> Dict[str, float]:
    ddpm_model.eval()
    inception_model_fid.eval()

    all_generated_samples_for_fid = [] # For FID, store normalized
    all_original_samples_for_fid = []  # For FID, store normalized

    # For MSE, MAE, MAPE, SMAPE, we operate on denormalized values
    all_generated_denorm_list = []
    all_original_denorm_list = []
    
    total_samples_processed_for_metrics = 0
    
    max_fid_samples = max_samples_for_fid if max_samples_for_fid is not None else len(dataloader.dataset)

    pbar = tqdm(dataloader, desc="Evaluating Model", leave=False)
    for batch_idx, (target_avg_flow_norm, hour_scalars, day_scalars, _) in enumerate(pbar): # _ is extra_data_rows
        current_batch_size = target_avg_flow_norm.shape[0]
        
        target_avg_flow_norm = target_avg_flow_norm.to(config["device"])
        
        generated_flow_norm = ddpm_model.sample(
            batch_size=current_batch_size,
            hour_scalars_batch=hour_scalars, # Passed as is, DDPM handles device
            day_scalars_batch=day_scalars   # Passed as is
        ) # Output is (N, 1, D, H, W), normalized

        # Denormalize for MSE/MAE/MAPE/SMAPE
        # Ensure norm_stats_flow is correctly accessed from the dataset instance
        if hasattr(dataloader.dataset, 'norm_stats_flow') and dataloader.dataset.norm_stats_flow is not None:
            mean_val = dataloader.dataset.norm_stats_flow['mean']
            std_val = dataloader.dataset.norm_stats_flow['std']
        else: # Fallback or error if norm_stats not found (should not happen if dataset is set up correctly)
            logger.error("Normalization stats (norm_stats_flow) not found in dataset. Cannot denormalize.")
            mean_val, std_val = 0, 1 # Default to no-op for denormalization if error

        generated_flow_denorm = generated_flow_norm * std_val + mean_val
        target_avg_flow_denorm = target_avg_flow_norm * std_val + mean_val
        
        all_generated_denorm_list.append(generated_flow_denorm.cpu())
        all_original_denorm_list.append(target_avg_flow_denorm.cpu())
        
        # For FID, use normalized samples
        if len(all_generated_samples_for_fid) * config["eval_batch_size"] < max_fid_samples :
             all_generated_samples_for_fid.append(generated_flow_norm.cpu()) 
             all_original_samples_for_fid.append(target_avg_flow_norm.cpu())
        
        total_samples_processed_for_metrics += current_batch_size
        if total_samples_processed_for_metrics >= max_fid_samples and batch_idx < (len(dataloader)-1) :
            # If collecting for FID and enough samples are gathered, can break early for FID part
            # But MSE/MAE/etc. should ideally run on the whole val/test set.
            # Let's assume max_fid_samples is also the limit for other metrics for simplicity here,
            # or remove this early break for MSE/MAE. For now, let it be.
            # logger.info(f"Collected enough samples ({total_samples_processed_for_metrics}) for FID. Breaking eval loop if only for FID.")
            pass # Continue to process all batches for aggregate metrics

    # Concatenate all denormalized batches for metric calculation
    if not all_generated_denorm_list: # Handle empty dataloader case
        logger.warning("No data processed during evaluation. Returning zero metrics.")
        return {"mse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0, "fid": -1.0}

    generated_all_denorm_t = torch.cat(all_generated_denorm_list, dim=0)
    original_all_denorm_t = torch.cat(all_original_denorm_list, dim=0)

    # Calculate metrics on all collected denormalized samples
    epsilon = 1e-8 # For MAPE/SMAPE to avoid division by zero
    
    mse_total = F.mse_loss(generated_all_denorm_t, original_all_denorm_t).item()
    mae_total = F.l1_loss(generated_all_denorm_t, original_all_denorm_t).item()
    
    # MAPE Calculation:
    # Be careful with target values being zero or very small.
    # Clamping target denominator for stability, or using a version robust to zeros.
    mape_total = torch.mean(torch.abs((original_all_denorm_t - generated_all_denorm_t) / 
                                     (torch.abs(original_all_denorm_t) + epsilon))) * 100
    mape_total = mape_total.item()

    # SMAPE Calculation (Common definition: 2 * |pred - actual| / (|actual| + |pred|) )
    smape_numerator = torch.abs(generated_all_denorm_t - original_all_denorm_t)
    smape_denominator = torch.abs(original_all_denorm_t) + torch.abs(generated_all_denorm_t) + epsilon
    smape_total = torch.mean(200 * smape_numerator / smape_denominator) # Multiply by 100 for percentage, and factor of 2
    smape_total = smape_total.item()

    metrics = {"mse": mse_total, "mae": mae_total, "mape": mape_total, "smape": smape_total, "fid": -1.0}

    # --- FID Calculation (uses normalized samples) ---
    actual_fid_samples_collected = len(all_generated_samples_for_fid) * (config["eval_batch_size"] if len(all_generated_samples_for_fid)>0 else 0) # This is wrong if batch size varies
    # Correct way: sum of bs, or len of concatenated tensor
    num_fid_samples_to_calc = 0
    if all_generated_samples_for_fid and all_original_samples_for_fid:
        generated_tensor_fid = torch.cat(all_generated_samples_for_fid, dim=0)[:max_fid_samples]
        original_tensor_fid = torch.cat(all_original_samples_for_fid, dim=0)[:max_fid_samples]
        num_fid_samples_to_calc = generated_tensor_fid.shape[0]

        if num_fid_samples_to_calc > 1 : # Need at least 2 samples for covariance matrix
            logger.info(f"Calculating FID on {num_fid_samples_to_calc} samples...")
            act_generated = get_activations(generated_tensor_fid, inception_model_fid, config["device"], config["fid_batch_size"])
            act_original = get_activations(original_tensor_fid, inception_model_fid, config["device"], config["fid_batch_size"])
            
            if act_generated.shape[0] > 1 and act_original.shape[0] > 1: # Ensure enough samples after get_activations
                 metrics["fid"] = calculate_fid(act_original, act_generated)
                 logger.info(f"FID Calculated: {metrics['fid']:.4f}")
            else:
                logger.warning("Not enough valid activations for FID calculation after processing.")
                metrics["fid"] = float('nan') # Or -1.0
        else:
            logger.warning(f"Not enough samples ({num_fid_samples_to_calc}) collected or available for FID calculation.")
            metrics["fid"] = float('nan') # Or -1.0
    else:
        logger.warning("Sample lists for FID are empty.")
        metrics["fid"] = float('nan')

    return metrics

# Main training script
# (Assumed to be defined as in the previous complete script,
#  but the part calling ddpm.p_losses needs to pass scalar hour/day)
if __name__ == '__main__':
    logger.info("==========================================================")
    logger.info("    STARTING DDPM TRAINING (extra_data no norm, flow norm)    ")
    logger.info("==========================================================")
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    try: full_df = pd.read_csv(CONFIG["data_path"]); logger.info(f"Loaded data: {CONFIG['data_path']}. Shape: {full_df.shape}")
    except Exception as e: logger.error(f"Error loading data: {e}. Exiting."); exit()

    total_len = len(full_df); train_len = int(CONFIG["train_split_ratio"]*total_len); val_len = int(CONFIG["val_split_ratio"]*total_len)
    if val_len <= 0 or (total_len - train_len - val_len) <= 0: raise ValueError("Train/Val/Test split error.")
    df_shuffled = full_df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    train_df, val_df, test_df = df_shuffled[:train_len], df_shuffled[train_len:train_len+val_len], df_shuffled[train_len+val_len:]
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    try:
        logger.info("Creating training dataset...")
        train_dataset = PeopleFlowDatasetCondition(train_df, CONFIG, mode='train')
        logger.info("Creating validation dataset...")
        val_dataset = PeopleFlowDatasetCondition(val_df, CONFIG, mode='val',
                                               average_flow_map_dict=train_dataset.average_flow_map_dict,
                                               norm_stats_flow=train_dataset.norm_stats_flow,
                                               sorted_flow_columns=train_dataset.sorted_flow_columns,
                                               grid_idx_to_rc_map=train_dataset.grid_idx_to_rc_map,
                                               processed_extra_columns=train_dataset.processed_extra_columns)
        logger.info("Creating test dataset...")
        test_dataset = PeopleFlowDatasetCondition(test_df, CONFIG, mode='test',
                                                average_flow_map_dict=train_dataset.average_flow_map_dict,
                                                norm_stats_flow=train_dataset.norm_stats_flow,
                                                sorted_flow_columns=train_dataset.sorted_flow_columns,
                                                grid_idx_to_rc_map=train_dataset.grid_idx_to_rc_map,
                                                processed_extra_columns=train_dataset.processed_extra_columns)
    except Exception as e: logger.error(f"Error creating datasets: {e}.", exc_info=True); exit()
        
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    logger.info("DataLoaders created.")

    logger.info("Initializing UNet3D model...")
    unet = UNet3D(CONFIG["image_channels"], CONFIG["base_channels_unet"], CONFIG["time_emb_dim"], CONFIG["condition_encode_dim"]).to(CONFIG["device"])
    logger.info("Initializing DDPM3D model...")
    ddpm = DDPM3D(unet, CONFIG["timesteps"], (CONFIG["D"],CONFIG["H"],CONFIG["W"]), CONFIG["image_channels"],
                  CONFIG["condition_input_channels"], CONFIG["condition_encode_dim"],
                  CONFIG["beta_start"], CONFIG["beta_end"], CONFIG["device"]).to(CONFIG["device"])
    logger.info(f"UNet3D params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    logger.info(f"ConditionProcessor params: {sum(p.numel() for p in ddpm.condition_processor.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(list(ddpm.model.parameters()) + list(ddpm.condition_processor.parameters()), lr=CONFIG["lr"])
    
    logger.info("Loading InceptionV3 for FID...")
    inception_fid = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
    inception_fid.fc = nn.Identity(); inception_fid = inception_fid.to(CONFIG["device"]); inception_fid.eval()
    logger.info("InceptionV3 loaded.")

    logger.info("Starting training loop...")
    best_val_metric = float('inf') # Using MSE for best model
    metrics_hist = {'train_loss':[],'val_loss':[],'val_mse':[],'val_mae':[],'val_fid':[]}

    for epoch in range(1, CONFIG["epochs"] + 1):
        ddpm.train(); total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [Train]", leave=False)
        for x_start, hour_s, day_s, _ in train_pbar: # _ is extra_data_rows
            optimizer.zero_grad()
            x_start = x_start.to(CONFIG["device"])
            # hour_s, day_s are (N,), already on CPU from dataset, DDPM methods will move to device
            t = torch.randint(0, CONFIG["timesteps"], (x_start.shape[0],), device=CONFIG["device"]).long()
            loss = ddpm.p_losses(x_start, t, hour_s, day_s) # Pass scalar hour/day
            loss.backward(); optimizer.step(); total_train_loss += loss.item()
            train_pbar.set_postfix({"Loss": loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        metrics_hist['train_loss'].append(avg_train_loss)

        logger.info(f"Epoch {epoch} - Validating...")
        val_metrics = evaluate_model(ddpm, val_loader, inception_fid, CONFIG, CONFIG["fid_num_samples"]//2)
        
        metrics_hist['val_loss'].append(val_metrics['mse']); metrics_hist['val_mse'].append(val_metrics['mse'])
        metrics_hist['val_mae'].append(val_metrics['mae']); metrics_hist['val_fid'].append(val_metrics['fid'])
        logger.info(f"E{epoch}: TrainL:{avg_train_loss:.5f}|ValMSE:{val_metrics['mse']:.5f}|ValMAE:{val_metrics['mae']:.5f}|ValFID:{val_metrics['fid']:.3f}")

        if val_metrics['mse'] < best_val_metric:
            best_val_metric = val_metrics['mse']
            save_path = os.path.join(CONFIG["save_dir"], "best_ddpm_model.pth")
            torch.save({
                'epoch': epoch, 'ddpm_state_dict': ddpm.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric, 'config': CONFIG,
                'norm_stats_flow': train_dataset.norm_stats_flow, # Save flow norm stats
                'sorted_flow_columns': train_dataset.sorted_flow_columns,
                'grid_idx_to_rc_map': train_dataset.grid_idx_to_rc_map,
                'processed_extra_columns': train_dataset.processed_extra_columns,
                # 'extra_cont_mean': train_dataset.extra_cont_mean, # Not saving as not used for norm
                # 'extra_cont_std': train_dataset.extra_cont_std,   # Not saving as not used for norm
            }, save_path)
            logger.info(f"Saved new best model to {save_path} (Val MSE: {best_val_metric:.5f})")
            
        if epoch % (CONFIG["epochs"]//5 if CONFIG["epochs"] >=5 else 1) == 0 or epoch == CONFIG["epochs"]:
            ddpm.eval()
            with torch.no_grad():
                fixed_x_s, fixed_hr_s, fixed_day_s, _ = next(iter(val_loader))
                num_viz = min(4, fixed_hr_s.shape[0])
                fixed_hr_s, fixed_day_s = fixed_hr_s[:num_viz], fixed_day_s[:num_viz]
                
                logger.info(f"Generating sample viz for epoch {epoch}...")
                gen_samples = ddpm.sample(num_viz, fixed_hr_s, fixed_day_s)
                
                mean_v, std_v = train_dataset.norm_stats_flow['mean'], train_dataset.norm_stats_flow['std']
                gen_denorm = gen_samples.cpu() * std_v + mean_v
                
                fig, axes = plt.subplots(2, num_viz, figsize=(num_viz*3,6.5), squeeze=False)
                for i in range(num_viz):
                    ax_orig, ax_gen = axes[0,i], axes[1,i]
                    orig_denorm = fixed_x_s[i].cpu()*std_v+mean_v
                    hr_title, dow_title = int(fixed_hr_s[i].item()), int(fixed_day_s[i].item())
                    
                    im_o = ax_orig.imshow(orig_denorm.squeeze().numpy(), cmap='viridis'); ax_orig.set_title(f"Target (H{hr_title} D{dow_title})"); ax_orig.axis('off'); fig.colorbar(im_o, ax=ax_orig, fraction=0.046, pad=0.04)
                    im_g = ax_gen.imshow(gen_denorm[i].squeeze().numpy(), cmap='viridis'); ax_gen.set_title(f"Generated (H{hr_title} D{dow_title})"); ax_gen.axis('off'); fig.colorbar(im_g, ax=ax_gen, fraction=0.046, pad=0.04)
                plt.tight_layout(); plt.savefig(os.path.join(CONFIG["save_dir"],f"epoch_{epoch:03d}_samples.png")); plt.close(fig)
                logger.info(f"Saved sample viz for epoch {epoch}.")
    logger.info("Training finished.")

    logger.info("Loading best model for final test set evaluation...")
    chkpt = torch.load(os.path.join(CONFIG["save_dir"], "best_ddpm_model.pth"), map_location=CONFIG["device"])
    cfg_chkpt = chkpt['config'] # Use config from checkpoint for model re-init
    final_unet = UNet3D(cfg_chkpt["image_channels"],cfg_chkpt["base_channels_unet"],cfg_chkpt["time_emb_dim"],cfg_chkpt["condition_encode_dim"]).to(CONFIG["device"])
    final_ddpm = DDPM3D(final_unet,cfg_chkpt["timesteps"],(cfg_chkpt["D"],cfg_chkpt["H"],cfg_chkpt["W"]),cfg_chkpt["image_channels"],
                        cfg_chkpt["condition_input_channels"],cfg_chkpt["condition_encode_dim"], device=CONFIG["device"]) # Use current device
    final_ddpm.load_state_dict(chkpt['ddpm_state_dict'])
    logger.info("Best model loaded.")

    test_metrics = evaluate_model(final_ddpm, test_loader, inception_fid, CONFIG, CONFIG["fid_num_samples"])
    logger.info(f"FINAL TEST: MSE:{test_metrics['mse']:.5f}|MAE:{test_metrics['mae']:.5f}|FID:{test_metrics['fid']:.3f}")
    with open(os.path.join(CONFIG["save_dir"], "final_test_metrics.json"),'w') as f: json.dump(test_metrics,f,indent=4)
    with open(os.path.join(CONFIG["save_dir"], "final_test_metrics.txt"),'w') as f:
        f.write(f"FINAL TEST METRICS (FID on {CONFIG['fid_num_samples']} samples):\nDate: {pd.Timestamp.now(tz='Asia/Taipei')}\n")
        for k,v in test_metrics.items(): f.write(f"{k.upper()}: {v:.6f}\n")

    try:
        ep_rng = range(1, len(metrics_hist['train_loss']) + 1)
        plt.figure(figsize=(18,5)); plt.style.use('seaborn_v0_8_darkgrid')
        plt.subplot(1,3,1); plt.plot(ep_rng,metrics_hist['train_loss'],label='Train Loss'); plt.plot(ep_rng,metrics_hist['val_mse'],label='Val MSE'); plt.legend(); plt.title('Loss/MSE'); plt.grid(True)
        plt.subplot(1,3,2); plt.plot(ep_rng,metrics_hist['val_mae'],label='Val MAE',color='orange'); plt.legend(); plt.title('Val MAE'); plt.grid(True)
        plt.subplot(1,3,3); plt.plot(ep_rng,metrics_hist['val_fid'],label='Val FID',color='green'); plt.legend(); plt.title('Val FID'); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(CONFIG["save_dir"],"training_history_plots.png")); plt.close()
        logger.info("Saved training history plots.")
    except Exception as e: logger.error(f"Error plotting history: {e}")

    logger.info("================ SCRIPT FINISHED ================")
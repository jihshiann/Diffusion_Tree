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
from tqdm import tqdm

# ==============================================================================
# 日誌設定
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 組態設定
# ==============================================================================
CONFIG = {
    # --- 資料參數 ---
    "data_path": "all_merged_sample.csv", # 資料路徑
    "H": 20, # 網格高度
    "W": 20, # 網格寬度
    "D": 1,  # 網格深度 (流量圖為1)

    # --- 模型參數 ---
    "image_channels": 1,      # 主要資料(流量圖)的通道數
    "condition_input_channels": 2, # 條件處理器接收的原始條件通道數 (小時網格 + 星期網格)
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 (可調)
    "base_channels_unet": 64,   # UNet3D 的基礎通道數
    "time_emb_dim": 256,        # 時間嵌入維度

    # --- DDPM 參數 ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 訓練參數 ---
    "epochs": 200, # 可調整
    "batch_size": 16, # 依 GPU 記憶體調整
    "lr": 1e-4, # 學習率
    "num_workers": 0, # DataLoader 工作執行緒 (Windows 建議 0, Linux 可 >0)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42, # 隨機種子

    # --- 評估參數 ---
    "eval_batch_size": 16,
    "fid_batch_size": 32,
    "fid_num_samples": 500, # FID 計算樣本數

    # --- 路徑與儲存 ---
    "save_dir": "results_ddpm_conditioned_flow_taipei_extra_v2", # 結果儲存目錄
    "plot_grid_mapping_path": "grid_mapping_visualization_taipei.png", # 網格映射視覺化圖片路徑
    "train_split_ratio": 0.7, # 訓練集比例 (新增)
    "val_split_ratio": 0.15,  # 驗證集比例 (新增)
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
# 資料集類別 (PeopleFlowDatasetCondition)
# ==============================================================================
class PeopleFlowDatasetCondition(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 config: Dict[str, Any],
                 mode: str = 'train',
                 # 驗證/測試模式下，由訓練資料集實例傳入
                 average_flow_map_dict: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                 norm_stats_flow: Optional[Dict[str, float]] = None, # 流量資料標準化用
                 sorted_flow_columns: Optional[List[str]] = None,
                 grid_idx_to_rc_map: Optional[Dict[int, Tuple[int,int]]] = None,
                 # extra_cont_mean, extra_cont_std 不再使用 (額外資料未標準化)
                 processed_extra_columns: Optional[List[str]] = None # 已處理額外資料欄位名稱
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

        # --- 時間解析 (從原始 DATATIME) ---
        if 'DATATIME' not in self.df_original.columns:
            raise ValueError("資料中未找到 'DATATIME' 欄位。")
        try:
            df_datetime_processed = self.df_original.copy()
            df_datetime_processed['DATATIME'] = pd.to_datetime(df_datetime_processed['DATATIME'])
            self.hours_original_np = df_datetime_processed['DATATIME'].dt.hour.values
            self.day_of_week_original_np = df_datetime_processed['DATATIME'].dt.dayofweek.values
        except Exception as e:
            raise ValueError(f"無法解析 'DATATIME' 欄位: {e}。")

        # --- 處理額外資料 (氣象、假日、時間特徵等) ---
        df_for_extra_processing = self.df_original.copy()

        temp_dt_series = pd.to_datetime(df_for_extra_processing['DATATIME'])
        df_for_extra_processing['年'] = temp_dt_series.dt.year
        df_for_extra_processing['月'] = temp_dt_series.dt.month
        df_for_extra_processing['日'] = temp_dt_series.dt.day
        df_for_extra_processing['時'] = temp_dt_series.dt.hour # 原始小時
        df_for_extra_processing['weekday'] = temp_dt_series.dt.dayofweek # 原始星期

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
        
        # 確保只選取實際存在的欄位
        present_extra_cols = [col for col in self.extra_cols_list_definition if col in df_for_extra_processing.columns]
        df_extra_subset = df_for_extra_processing[present_extra_cols].copy()


        if "hoilday" in df_extra_subset.columns: # 修正可能的錯字
            df_extra_subset.rename(columns={"hoilday": "holiday"}, inplace=True)

        cat_features = ['holiday'] # 分類特徵
        actual_cat_features = [col for col in cat_features if col in df_extra_subset.columns]

        if actual_cat_features:
            df_extra_subset[actual_cat_features] = df_extra_subset[actual_cat_features].astype(str)
            df_cat = pd.get_dummies(df_extra_subset[actual_cat_features], prefix=actual_cat_features, dummy_na=False)
        else:
            df_cat = pd.DataFrame(index=df_extra_subset.index)

        df_cont = df_extra_subset.drop(columns=actual_cat_features, errors='ignore')

        # 連續額外特徵不進行標準化
        df_extra_processed = pd.concat([df_cont, df_cat], axis=1) # 使用原始連續 + one-hot 分類

        if self.mode == 'train':
            self.processed_extra_columns = list(df_extra_processed.columns)
            self.processed_extra_data_np = df_extra_processed.fillna(0).values.astype(np.float32)
            logger.info(f"訓練集: 已處理 {len(self.processed_extra_columns)} 個額外特徵 (連續特徵未標準化)。")
        else:
            if processed_extra_columns is None:
                raise ValueError("驗證/測試模式，必須提供 processed_extra_columns。")
            self.processed_extra_columns = processed_extra_columns
            # 確保欄位順序一致並填補缺失值 (例如 one-hot)
            df_extra_processed = df_extra_processed.reindex(columns=self.processed_extra_columns, fill_value=0)
            self.processed_extra_data_np = df_extra_processed.fillna(0).values.astype(np.float32)
            logger.info(f"{self.mode} 資料集: 已處理額外特徵 (連續特徵未標準化)。")

        # --- 流量資料網格映射與平均值計算 ---
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
            self.flow_mean_val = np.mean(all_avg_flows_np) # 流量資料標準化用
            self.flow_std_val = np.std(all_avg_flows_np)   # 流量資料標準化用
            if self.flow_std_val < 1e-5: self.flow_std_val = 1e-5 # 避免除以零
            self.norm_stats_flow = {'mean': self.flow_mean_val, 'std': self.flow_std_val}
            logger.info(f"訓練集流量標準化統計量: 平均值={self.flow_mean_val:.4f}, 標準差={self.flow_std_val:.4f}")

        else: # 驗證或測試模式
            if not all([average_flow_map_dict, norm_stats_flow, sorted_flow_columns, grid_idx_to_rc_map]):
                raise ValueError("average_flow_map_dict, norm_stats_flow, sorted_flow_columns, grid_idx_to_rc_map 必須為驗證/測試模式提供。")
            self.average_flow_map_dict = average_flow_map_dict
            self.norm_stats_flow = norm_stats_flow # 使用傳入的流量標準化統計量
            self.flow_mean_val = self.norm_stats_flow['mean']
            self.flow_std_val = self.norm_stats_flow['std']
            self.sorted_flow_columns = sorted_flow_columns
            self.grid_idx_to_rc_map = grid_idx_to_rc_map
            logger.info(f"使用預計算的流量標準化統計量: 平均值={self.flow_mean_val:.4f}, 標準差={self.flow_std_val:.4f}")

    # 輔助方法 _extract_all_sensor_info_from_csv, _select_sensors,
    # _define_target_grid_cells_hierarchical_style, _map_sensors_to_target_grid_hungarian,
    # _calculate_average_flows, _plot_grid_mapping 與先前版本相同。
    # 為求類別完整性，再次貼上。

    def _extract_all_sensor_info_from_csv(self) -> List[Dict[str, Any]]:
        """從 CSV 欄位名稱提取所有感測器資訊 (名稱、經緯度)"""
        all_sensor_info = []
        max_sensor_idx = -1
        # 從 flow_i, latitude_i, longitude_i 欄位找出最大索引 i
        for col in self.df_original.columns:
            if col.startswith('flow_') or col.startswith('latitude_') or col.startswith('longitude_'):
                try:
                    idx = int(col.split('_')[-1])
                    max_sensor_idx = max(max_sensor_idx, idx)
                except ValueError:
                    continue # 若無法解析索引則跳過
        if max_sensor_idx == -1:
            raise ValueError("無法從 CSV 欄位名確定感測器索引。")

        for i in range(max_sensor_idx + 1):
            fcol, latcol, loncol = f'flow_{i}', f'latitude_{i}', f'longitude_{i}'
            if all(c in self.df_original.columns for c in [fcol, latcol, loncol]): # 檢查欄位是否存在
                # 取第一筆非空的經緯度資料作為此感測器的位置
                lat_series = self.df_original[latcol].dropna()
                lon_series = self.df_original[loncol].dropna()
                if not lat_series.empty and not lon_series.empty:
                    lat, lon = lat_series.iloc[0], lon_series.iloc[0]
                    all_sensor_info.append({'name': fcol, 'lon': float(lon), 'lat': float(lat), 'original_csv_sensor_index': i})
        if not all_sensor_info:
            raise ValueError("無法從 CSV 提取有效感測器資料。")
        return all_sensor_info

    def _select_sensors(self, all_sensor_info: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """根據網格數量選取感測器。若感測器過多，選取最接近地理中心者。"""
        num_required = self.num_grid_cells
        if len(all_sensor_info) < num_required:
            raise ValueError(f"網格點數 ({num_required}) > 可用感測器 ({len(all_sensor_info)})")

        coords = np.array([(s['lon'], s['lat']) for s in all_sensor_info])
        if len(all_sensor_info) > num_required:
            logger.info(f"可用感測器 ({len(all_sensor_info)}) > 所需 ({num_required})。選取最接近中心的點。")
            center = np.mean(coords, axis=0) # 計算所有感測器的地理中心
            dists = np.sum((coords - center)**2, axis=1) # 計算各感測器到中心的距離平方
            sel_indices = np.argsort(dists)[:num_required] # 選取距離最近的 num_required 個感測器
            sel_info = [all_sensor_info[i] for i in sel_indices]
            sel_coords = coords[sel_indices]
        else: # 感測器數量剛好或不足 (已在前一步檢查不足情況)
            sel_info = all_sensor_info
            sel_coords = coords
        logger.info(f"已選定 {len(sel_info)} 個感測器進行網格映射。")
        return sel_info, sel_coords

    def _define_target_grid_cells_hierarchical_style(self, selected_real_coords_np: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
        """以 'hierarchical' 風格定義目標網格中心點。"""
        if selected_real_coords_np.shape[0] != self.num_grid_cells:
            raise ValueError(f"selected_real_coords_np 應含 {self.num_grid_cells} 點, 得到 {selected_real_coords_np.shape[0]}。")
        logger.info("使用 'hierarchical' 風格定義目標網格中心點...")
        center_lon, center_lat = np.mean(selected_real_coords_np, axis=0) # 選定感測器的中心
        unique_lons = np.unique(selected_real_coords_np[:,0])
        unique_lats = np.unique(selected_real_coords_np[:,1])
        lon_diffs = np.diff(np.sort(unique_lons))
        lat_diffs = np.diff(np.sort(unique_lats))
        # 使用差值的中位數作為步長，忽略過小差值
        lon_step = np.median(lon_diffs[lon_diffs > 1e-6]) if len(lon_diffs[lon_diffs > 1e-6]) > 0 else 0.001
        lat_step = np.median(lat_diffs[lat_diffs > 1e-6]) if len(lat_diffs[lat_diffs > 1e-6]) > 0 else 0.001
        if lon_step <= 1e-6: lon_step = 0.001 # 預設最小步長
        if lat_step <= 1e-6: lat_step = 0.001 # 預設最小步長
        logger.info(f"目標網格中心: (lon:{center_lon:.4f}, lat:{center_lat:.4f}), 步長: (lon:{lon_step:.6f}, lat:{lat_step:.6f})")

        grid_targets = np.zeros((self.num_grid_cells, 2)) # (H*W, 2) 存放目標網格的經緯度
        idx_to_rc = {} # 將網格平面索引映射到 (row, col)
        idx = 0
        for r_idx in range(self.H):
            for c_idx in range(self.W):
                # 以中心點和步長計算網格點座標
                tlon = center_lon + (c_idx - (self.W - 1) / 2.0) * lon_step
                tlat = center_lat - (r_idx - (self.H - 1) / 2.0) * lat_step # 緯度通常由北往南增加索引
                grid_targets[idx, 0], grid_targets[idx, 1] = tlon, tlat
                idx_to_rc[idx] = (r_idx, c_idx)
                idx += 1
        return grid_targets, idx_to_rc

    def _map_sensors_to_target_grid_hungarian(self, sel_info: List[Dict[str,Any]], sel_coords: np.ndarray, grid_targets: np.ndarray) -> List[str]:
        """使用匈牙利演算法將選定感測器映射到目標網格。"""
        logger.info("使用匈牙利演算法將選定感測器映射到目標網格...")
        n = self.num_grid_cells
        if not (sel_coords.shape[0] == n and grid_targets.shape[0] == n and len(sel_info) == n):
            raise ValueError("匈牙利分配時，輸入數量必須都等於 H*W。")
        # 計算成本矩陣：實際感測器位置到目標網格中心的歐氏距離
        costs = np.sum((sel_coords[:, np.newaxis, :] - grid_targets[np.newaxis, :, :])**2, axis=2)
        costs = np.sqrt(costs)
        real_indices, target_indices = linear_sum_assignment(costs) # 匈牙利演算法指派

        target_to_real_map = {t_idx: r_idx for r_idx, t_idx in zip(real_indices, target_indices)}
        sorted_cols = [""] * n # 存放排序後的流量欄位名稱
        for flat_target_idx in range(n): # flat_target_idx 是目標網格的平面索引 (0 to H*W-1)
            if flat_target_idx in target_to_real_map:
                real_idx_in_sel_list = target_to_real_map[flat_target_idx] # 取得分配到的實際感測器在 sel_info 中的索引
                sorted_cols[flat_target_idx] = sel_info[real_idx_in_sel_list]['name']
            else:
                # 理論上匈牙利演算法會為每個目標找到一個匹配 (如果數量相等)
                raise Exception(f"目標網格 {flat_target_idx} 未分配到感測器。")
        logger.info(f"成功為網格排序 {len(sorted_cols)} 個流量欄位。")
        return sorted_cols

    def _calculate_average_flows(self) -> Dict[Tuple[int, int], np.ndarray]:
        """計算每個 (小時, 星期幾) 組合的平均流量圖。"""
        logger.info("計算 (小時, 星期幾) 平均流量圖...")
        avg_flows = {} # (小時, 星期幾) -> (H, W) 平均流量陣列
        for col in self.sorted_flow_columns:
            if col not in self.df_original.columns:
                raise ValueError(f"流量欄位 '{col}' 在 DataFrame 未找到。")

        # 取得排序後的流量資料，並轉換為 (時間點數量, H*W) 的 NumPy 陣列
        flow_data_grid_alltimes = self.df_original[self.sorted_flow_columns].values.astype(np.float32)

        grouping_df = pd.DataFrame({'hour': self.hours_original_np, 'day_of_week': self.day_of_week_original_np})

        for (hr, dow), group_indices in grouping_df.groupby(['hour', 'day_of_week']).groups.items():
            group_flows = flow_data_grid_alltimes[group_indices] # 取得該 (小時, 星期幾) 的所有流量資料
            mean_flow_flat = np.nanmean(group_flows, axis=0) # 沿時間軸計算平均，忽略 NaN
            mean_flow_flat[np.isnan(mean_flow_flat)] = 0 # 將剩餘 NaN (若某網格點一直都是 NaN) 設為 0
            avg_flows[(hr, dow)] = mean_flow_flat.reshape(self.H, self.W)
        if not avg_flows:
            logger.warning("未計算任何平均流量。")
        logger.info(f"計算完成 {len(avg_flows)} 個條件的平均流量。")
        return avg_flows

    def _plot_grid_mapping(self, sel_coords, grid_targets, idx_to_rc, sorted_cols, save_path):
        """繪製並儲存網格映射的視覺化圖。"""
        try:
            plt.figure(figsize=(10,10)); plt.style.use('seaborn_v0_8_whitegrid')
            plt.scatter(sel_coords[:,0], sel_coords[:,1], c='blue', marker='o', s=25, alpha=0.7, label='選定實際感測器位置', zorder=2)
            plt.scatter(grid_targets[:,0], grid_targets[:,1], c='red', marker='x', s=25, alpha=0.7, label='目標網格中心點', zorder=3)
            for flat_idx in range(self.num_grid_cells):
                r_idx, c_idx = idx_to_rc.get(flat_idx, (-1,-1))
                if r_idx != -1:
                    plt.text(grid_targets[flat_idx,0], grid_targets[flat_idx,1], f'T[{r_idx},{c_idx}]', fontsize=5,color='darkred',ha='center',va='bottom',zorder=4)
            plt.xlabel("經度 (台北市)"); plt.ylabel("緯度"); plt.title(f"網格映射 ({self.H}x{self.W})")
            plt.legend(); plt.savefig(save_path, dpi=200); plt.close()
            logger.info(f"網格映射視覺化圖儲存至 {save_path}")
        except Exception as e:
            logger.error(f"繪製網格圖出錯: {e}", exc_info=True)

    def __len__(self) -> int:
        return len(self.df_original)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        # --- 目標平均流量 (DDPM 的 x_start) - 已標準化 ---
        current_hour_original = self.hours_original_np[idx]
        current_dow_original = self.day_of_week_original_np[idx]

        target_avg_flow_np = self.average_flow_map_dict.get((current_hour_original, current_dow_original))
        if target_avg_flow_np is None: # 若特定 (時,星期) 無平均流量 (罕見)，則用零填充
            logger.warning(f"在 average_flow_map_dict 中找不到 ({current_hour_original}, {current_dow_original}) 的平均流量，使用零填充。")
            target_avg_flow_np = np.zeros((self.H, self.W), dtype=np.float32)

        # 標準化流量資料
        standardized_avg_flow_np = (target_avg_flow_np - self.flow_mean_val) / self.flow_std_val
        target_flow_tensor = torch.from_numpy(standardized_avg_flow_np).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

        # --- 條件輸入: 原始小時 (0-23) 與星期 (0-6) ---
        # DDPM3D 內部會將其轉換為正規化網格

        # --- 已處理的額外資料列 (連續特徵未標準化) ---
        extra_data_row_tensor = torch.from_numpy(self.processed_extra_data_np[idx]).float() # (num_processed_extra_features,)

        return target_flow_tensor, int(current_hour_original), int(current_dow_original), extra_data_row_tensor

# ==============================================================================
# UNet3D, DDPM3D, FID, 評估及主訓練迴圈
# (UNet3D, SinusoidalTimeEmbedding, DoubleConv3D, Down3D, Up3D, OutConv3D
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
    """3D U-Net 模型"""
    def __init__(self, input_image_channels: int, base_channels: int = 64, time_emb_dim: int = 256,
                 condition_encode_dim: Optional[int] = None, bilinear_upsample: bool = True):
        super().__init__()
        self.input_image_channels = input_image_channels
        self.condition_encode_dim = condition_encode_dim or 0 # 若為 None 則設為 0

        # 時間嵌入 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        actual_in_channels = self.input_image_channels + self.condition_encode_dim
        self.inc = DoubleConv3D(actual_in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels*2)
        self.down2 = Down3D(base_channels*2, base_channels*4)
        self.down3 = Down3D(base_channels*4, base_channels*8)
        factor = 2 if bilinear_upsample else 1
        self.down4 = Down3D(base_channels*8, base_channels*16 // factor) # 最底層

        self.up1 = Up3D(base_channels*16, base_channels*8 // factor, bilinear_upsample)
        self.up2 = Up3D(base_channels*8, base_channels*4 // factor, bilinear_upsample)
        self.up3 = Up3D(base_channels*4, base_channels*2 // factor, bilinear_upsample)
        self.up4 = Up3D(base_channels*2, base_channels, bilinear_upsample)
        self.outc = OutConv3D(base_channels, self.input_image_channels) # 輸出通道數同影像通道數

    def _add_time_embedding(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """將時間嵌入加到特徵圖上"""
        # t_emb: (N, time_emb_dim) -> (N, time_emb_dim, 1, 1, 1)
        t_emb_expanded = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x + t_emb_expanded # 廣播加法

    def forward(self, x_t: torch.Tensor, time_steps: torch.Tensor, processed_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_t: (N, C_img, D, H, W)
        # time_steps: (N,)
        # processed_condition: (N, C_cond_enc, D, H, W)
        t_emb = self.time_mlp(time_steps) # (N, time_emb_dim)

        if processed_condition is not None:
            if x_t.shape[2:] != processed_condition.shape[2:]: # 檢查 D, H, W 是否一致
                raise ValueError(f"x_t DHW {x_t.shape[2:]} != processed_condition DHW {processed_condition.shape[2:]}")
            x_input = torch.cat((x_t, processed_condition), dim=1) # 沿通道維度合併
        else: # 無條件或條件已整合
            x_input = x_t

        x1 = self.inc(x_input);  x1 = self._add_time_embedding(x1, t_emb)
        x2 = self.down1(x1);   x2 = self._add_time_embedding(x2, t_emb)
        x3 = self.down2(x2);   x3 = self._add_time_embedding(x3, t_emb)
        x4 = self.down3(x3);   x4 = self._add_time_embedding(x4, t_emb)
        x5 = self.down4(x4);   x5 = self._add_time_embedding(x5, t_emb) # Bottleneck

        x = self.up1(x5, x4);  x = self._add_time_embedding(x, t_emb)
        x = self.up2(x, x3);  x = self._add_time_embedding(x, t_emb)
        x = self.up3(x, x2);  x = self._add_time_embedding(x, t_emb)
        x = self.up4(x, x1);  x = self._add_time_embedding(x, t_emb)
        return self.outc(x) # 預測雜訊

def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """線性 beta 排程"""
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM3D(nn.Module):
    """3D Denoising Diffusion Probabilistic Model"""
    def __init__(self,
                 unet_model: UNet3D,
                 timesteps: int,
                 image_size: Tuple[int, int, int], # (D, H, W)
                 image_channels: int,
                 condition_input_channels: int, # 條件處理器輸入的原始通道數 (例如: 小時網格+星期網格 = 2)
                 condition_encode_dim: int,     # 條件處理器輸出的編碼維度
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device: str = "cpu"):
        super().__init__()
        self.model = unet_model # U-Net 模型
        self.timesteps = timesteps
        self.image_size_D, self.image_size_H, self.image_size_W = image_size # 儲存 D, H, W
        self.image_channels = image_channels
        self.device = device

        # --- 擴散排程參數 ---
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # α_bar_t
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # α_bar_{t-1}
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) # p(x_{t-1}|x_t, x_0) 的變異數

        # --- 條件處理器 (例如：將小時、星期幾網格編碼) ---
        # 輸入: (N, condition_input_channels, D, H, W)
        # 輸出: (N, condition_encode_dim, D, H, W)
        self.condition_processor = nn.Sequential(
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False), # 深度維度 kernel=1, padding=0
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False), # 深度維度 kernel=1, padding=0
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()
        ).to(device)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """從 a 中提取對應 t 時刻的值，並調整形狀以匹配 x_shape"""
        batch_size = t.shape[0]
        out = a.gather(-1, t) # (batch_size,)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))) # (batch_size, 1, 1, 1, 1)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向擴散過程 (加噪)：q(x_t | x_0)"""
        if noise is None: noise = torch.randn_like(x_start)
        sact = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        soma_ct = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sact * x_start + soma_ct * noise # x_t

    def _prepare_conditional_input_grids(self,
                                        hour_scalars_batch: torch.Tensor, # (N,) 原始 0-23
                                        day_scalars_batch: torch.Tensor,  # (N,) 原始 0-6
                                        ) -> torch.Tensor: # 輸出 (N, 2, D, H, W)
        """將純量的小時和星期幾轉換為正規化的網格輸入"""
        batch_size = hour_scalars_batch.shape[0]
        # 在此正規化純量值
        norm_hours = hour_scalars_batch.float().to(self.device) / 23.0 # 正規化到 [0, 1]
        norm_days = day_scalars_batch.float().to(self.device) / 6.0   # 正規化到 [0, 1]

        # 建立 HxW 的網格，每個網格的值相同
        hour_grids_list = [torch.full((self.image_size_H, self.image_size_W), norm_hours[i].item(), device=self.device, dtype=torch.float32) for i in range(batch_size)]
        day_grids_list = [torch.full((self.image_size_H, self.image_size_W), norm_days[i].item(), device=self.device, dtype=torch.float32) for i in range(batch_size)]

        hour_grids_t = torch.stack(hour_grids_list, dim=0).unsqueeze(1).unsqueeze(2) # (N,1,1,H,W)
        day_grids_t = torch.stack(day_grids_list, dim=0).unsqueeze(1).unsqueeze(2)   # (N,1,1,H,W)

        # 確保深度維度匹配 self.image_size_D (在此專案中 D=1)
        if self.image_size_D != 1: # 理論上此專案不會進入此分支
             hour_grids_t = hour_grids_t.repeat(1,1,self.image_size_D,1,1)
             day_grids_t = day_grids_t.repeat(1,1,self.image_size_D,1,1)

        return torch.cat((hour_grids_t, day_grids_t), dim=1) # (N, 2, D, H, W)

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor,
                 hour_scalars_batch: torch.Tensor, day_scalars_batch: torch.Tensor,
                 # extra_data_batch: torch.Tensor, # 目前未直接由條件處理器使用
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """計算損失 (預測雜訊與真實雜訊的 MSE)"""
        if noise is None: noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise) # 得到加噪影像 x_t

        # 準備並處理條件輸入
        stacked_cond_grids = self._prepare_conditional_input_grids(hour_scalars_batch, day_scalars_batch) # (N, 2, D, H, W)
        processed_condition = self.condition_processor(stacked_cond_grids) # (N, C_cond_enc, D, H, W)

        predicted_noise = self.model(x_t, t, processed_condition) # U-Net 預測雜訊
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_scalar: int, t_tensor_batch: torch.Tensor,
                 processed_conditions_batch: torch.Tensor) -> torch.Tensor:
        """逆向過程單步取樣：p(x_{t-1} | x_t)"""
        # t_tensor_batch 是 (batch_size,)，每個元素都是 t_scalar
        betas_t = self._extract(self.betas, t_tensor_batch, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor_batch, x_t.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t_tensor_batch, x_t.shape) # 1/sqrt(α_t)

        # 使用 U-Net 預測雜訊
        predicted_noise = self.model(x_t, t_tensor_batch, processed_conditions_batch)
        # 計算 x_0_hat 的均值部分 (DDPM 公式)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_scalar == 0: # 最後一步，直接返回均值
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_tensor_batch, x_t.shape)
            noise = torch.randn_like(x_t) # 加入隨機雜訊
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int,...], hour_scalars_batch: torch.Tensor, day_scalars_batch: torch.Tensor) -> torch.Tensor:
        """逆向過程完整取樣迴圈"""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device) # 從純雜訊 x_T 開始

        # 預先處理條件，因為在迴圈中條件是固定的
        stacked_cond_grids = self._prepare_conditional_input_grids(hour_scalars_batch, day_scalars_batch)
        processed_conditions = self.condition_processor(stacked_cond_grids) # (batch_size, C_cond_enc, D, H, W)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="DDPM 取樣迴圈", total=self.timesteps, leave=False):
            t_tensor_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, i, t_tensor_batch, processed_conditions)
        return img # 返回生成的影像 x_0

    @torch.no_grad()
    def sample(self, batch_size: int, hour_scalars_batch: torch.Tensor, day_scalars_batch: torch.Tensor) -> torch.Tensor:
        """生成一批樣本"""
        # hour_scalars_batch, day_scalars_batch 應為 (batch_size,)
        s = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        return self.p_sample_loop(s, hour_scalars_batch, day_scalars_batch)

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


# evaluate_model 函數
# (假設其定義與先前完整腳本相同，
# 但需傳遞純量小時/星期至 ddpm_model.sample)
@torch.no_grad()
def evaluate_model(ddpm_model: DDPM3D,
                   dataloader: DataLoader,
                   inception_model_fid: nn.Module,
                   config: Dict[str, Any],
                   max_samples_for_fid: Optional[int] = None # FID 計算的最大樣本數
                   ) -> Dict[str, float]:
    """評估 DDPM 模型，計算 MSE, MAE, MAPE, SMAPE, FID。"""
    ddpm_model.eval()
    inception_model_fid.eval()

    all_generated_samples_for_fid = [] # 儲存正規化的生成樣本 (FID用)
    all_original_samples_for_fid = []  # 儲存正規化的原始樣本 (FID用)

    # MSE, MAE, MAPE, SMAPE 在反正規化後的數值上操作
    all_generated_denorm_list = []
    all_original_denorm_list = []

    total_samples_processed_for_metrics = 0

    # 若未指定 FID 樣本數，則使用整個資料集
    max_fid_samples = max_samples_for_fid if max_samples_for_fid is not None else len(dataloader.dataset)

    pbar = tqdm(dataloader, desc="評估模型", leave=False)
    for batch_idx, (target_avg_flow_norm, hour_scalars, day_scalars, _) in enumerate(pbar): # _ 是 extra_data_rows
        current_batch_size = target_avg_flow_norm.shape[0]

        target_avg_flow_norm = target_avg_flow_norm.to(config["device"])

        # 生成流量圖 (正規化)
        generated_flow_norm = ddpm_model.sample(
            batch_size=current_batch_size,
            hour_scalars_batch=hour_scalars, # 直接傳遞，DDPM 內部處理裝置
            day_scalars_batch=day_scalars   # 直接傳遞
        ) # 輸出為 (N, 1, D, H, W)，已正規化

        # 反正規化以計算 MSE/MAE/MAPE/SMAPE
        if hasattr(dataloader.dataset, 'norm_stats_flow') and dataloader.dataset.norm_stats_flow is not None:
            mean_val = dataloader.dataset.norm_stats_flow['mean']
            std_val = dataloader.dataset.norm_stats_flow['std']
        else: # 若找不到標準化統計量 (理論上不應發生)
            logger.error("在資料集中找不到標準化統計量 (norm_stats_flow)。無法反正規化。")
            mean_val, std_val = 0, 1 # 預設為無操作

        generated_flow_denorm = generated_flow_norm * std_val + mean_val
        target_avg_flow_denorm = target_avg_flow_norm * std_val + mean_val

        all_generated_denorm_list.append(generated_flow_denorm.cpu())
        all_original_denorm_list.append(target_avg_flow_denorm.cpu())

        # 為 FID 收集正規化樣本
        if len(all_generated_samples_for_fid) * config.get("eval_batch_size", current_batch_size) < max_fid_samples : # 確保不超出 FID 樣本限制
             all_generated_samples_for_fid.append(generated_flow_norm.cpu())
             all_original_samples_for_fid.append(target_avg_flow_norm.cpu())

        total_samples_processed_for_metrics += current_batch_size
        # 此處不提早中斷，以便 MSE/MAE 等指標能在完整驗證/測試集上計算
        # FID 樣本數的限制主要影響 FID 計算部分

    # 串接所有反正規化的批次以計算指標
    if not all_generated_denorm_list: # 處理 dataloader 為空的情況
        logger.warning("評估期間未處理任何資料。返回零指標。")
        return {"mse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0, "fid": float('nan')} # FID 為 NaN

    generated_all_denorm_t = torch.cat(all_generated_denorm_list, dim=0)
    original_all_denorm_t = torch.cat(all_original_denorm_list, dim=0)

    # 在所有收集到的反正規化樣本上計算指標
    epsilon = 1e-8 # 用於 MAPE/SMAPE 避免除以零

    mse_total = F.mse_loss(generated_all_denorm_t, original_all_denorm_t).item()
    mae_total = F.l1_loss(generated_all_denorm_t, original_all_denorm_t).item()

    # MAPE 計算
    mape_total = torch.mean(torch.abs((original_all_denorm_t - generated_all_denorm_t) /
                                     (torch.abs(original_all_denorm_t) + epsilon))) * 100
    mape_total = mape_total.item()

    # SMAPE 計算 (常見定義: 200 * |pred - actual| / (|actual| + |pred| + epsilon))
    smape_numerator = torch.abs(generated_all_denorm_t - original_all_denorm_t)
    smape_denominator = torch.abs(original_all_denorm_t) + torch.abs(generated_all_denorm_t) + epsilon
    smape_total = torch.mean(200 * smape_numerator / smape_denominator)
    smape_total = smape_total.item()

    metrics = {"mse": mse_total, "mae": mae_total, "mape": mape_total, "smape": smape_total, "fid": float('nan')}

    # --- FID 計算 (使用正規化樣本) ---
    num_fid_samples_to_calc = 0
    if all_generated_samples_for_fid and all_original_samples_for_fid:
        generated_tensor_fid = torch.cat(all_generated_samples_for_fid, dim=0)[:max_fid_samples]
        original_tensor_fid = torch.cat(all_original_samples_for_fid, dim=0)[:max_fid_samples]
        num_fid_samples_to_calc = min(generated_tensor_fid.shape[0], original_tensor_fid.shape[0]) # 取實際收集到的較小者

        if num_fid_samples_to_calc > 1 : # 共變異數矩陣至少需要 2 個樣本
            logger.info(f"在 {num_fid_samples_to_calc} 個樣本上計算 FID...")
            act_generated = get_activations(generated_tensor_fid[:num_fid_samples_to_calc], inception_model_fid, config["device"], config["fid_batch_size"])
            act_original = get_activations(original_tensor_fid[:num_fid_samples_to_calc], inception_model_fid, config["device"], config["fid_batch_size"])

            if act_generated.shape[0] > 1 and act_original.shape[0] > 1: # 確保 get_activations 後仍有足夠樣本
                 metrics["fid"] = calculate_fid(act_original, act_generated)
                 logger.info(f"FID 計算完成: {metrics['fid']:.4f}")
            else:
                logger.warning("處理後，FID 計算的有效特徵不足。")
                metrics["fid"] = float('nan')
        else:
            logger.warning(f"收集或可用的 FID 計算樣本 ({num_fid_samples_to_calc}) 不足。")
            metrics["fid"] = float('nan')
    else:
        logger.warning("FID 的樣本列表為空。")
        metrics["fid"] = float('nan')

    return metrics

# 主訓練腳本
# (假設其定義與先前完整腳本相同，
# 但呼叫 ddpm.p_losses 的部分需傳遞純量小時/星期)
if __name__ == '__main__':
    logger.info("==========================================================")
    logger.info("    開始 DDPM 訓練 (額外資料未正規化，流量資料正規化)    ")
    logger.info("==========================================================")
    logger.info(f"組態設定: {json.dumps(CONFIG, indent=2)}")

    try:
        full_df = pd.read_csv(CONFIG["data_path"])
        logger.info(f"已載入資料: {CONFIG['data_path']}. 形狀: {full_df.shape}")
    except Exception as e:
        logger.error(f"載入資料錯誤: {e}. 程式結束。"); exit()

    # 資料分割
    total_len = len(full_df)
    train_len = int(CONFIG["train_split_ratio"] * total_len)
    val_len = int(CONFIG["val_split_ratio"] * total_len)
    test_len = total_len - train_len - val_len

    if train_len <= 0 or val_len <= 0 or test_len <= 0:
        raise ValueError(f"訓練/驗證/測試集分割錯誤。請檢查比例設定。"
                         f"Train: {train_len}, Val: {val_len}, Test: {test_len} out of {total_len}")

    df_shuffled = full_df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    train_df = df_shuffled[:train_len]
    val_df = df_shuffled[train_len : train_len + val_len]
    test_df = df_shuffled[train_len + val_len :]
    logger.info(f"資料分割: 訓練集={len(train_df)}, 驗證集={len(val_df)}, 測試集={len(test_df)}")


    try:
        logger.info("建立訓練資料集...")
        train_dataset = PeopleFlowDatasetCondition(train_df, CONFIG, mode='train')
        logger.info("建立驗證資料集...")
        val_dataset = PeopleFlowDatasetCondition(val_df, CONFIG, mode='val',
                                               average_flow_map_dict=train_dataset.average_flow_map_dict,
                                               norm_stats_flow=train_dataset.norm_stats_flow,
                                               sorted_flow_columns=train_dataset.sorted_flow_columns,
                                               grid_idx_to_rc_map=train_dataset.grid_idx_to_rc_map,
                                               processed_extra_columns=train_dataset.processed_extra_columns)
        logger.info("建立測試資料集...")
        test_dataset = PeopleFlowDatasetCondition(test_df, CONFIG, mode='test',
                                                average_flow_map_dict=train_dataset.average_flow_map_dict,
                                                norm_stats_flow=train_dataset.norm_stats_flow,
                                                sorted_flow_columns=train_dataset.sorted_flow_columns,
                                                grid_idx_to_rc_map=train_dataset.grid_idx_to_rc_map,
                                                processed_extra_columns=train_dataset.processed_extra_columns)
    except Exception as e:
        logger.error(f"建立資料集錯誤: {e}。", exc_info=True); exit()

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    logger.info("DataLoaders 建立完成。")

    logger.info("初始化 UNet3D 模型...")
    unet = UNet3D(CONFIG["image_channels"], CONFIG["base_channels_unet"], CONFIG["time_emb_dim"], CONFIG["condition_encode_dim"]).to(CONFIG["device"])
    logger.info("初始化 DDPM3D 模型...")
    ddpm = DDPM3D(unet, CONFIG["timesteps"], (CONFIG["D"],CONFIG["H"],CONFIG["W"]), CONFIG["image_channels"],
                  CONFIG["condition_input_channels"], CONFIG["condition_encode_dim"],
                  CONFIG["beta_start"], CONFIG["beta_end"], CONFIG["device"]).to(CONFIG["device"])
    logger.info(f"UNet3D 參數數量: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    logger.info(f"ConditionProcessor 參數數量: {sum(p.numel() for p in ddpm.condition_processor.parameters() if p.requires_grad):,}")

    # 優化器包含 U-Net 和條件處理器的參數
    optimizer = optim.AdamW(list(ddpm.model.parameters()) + list(ddpm.condition_processor.parameters()), lr=CONFIG["lr"])

    logger.info("載入 InceptionV3 以計算 FID...")
    inception_fid = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
    inception_fid.fc = nn.Identity() # 移除最後的全連接層以獲取特徵
    inception_fid = inception_fid.to(CONFIG["device"])
    inception_fid.eval()
    logger.info("InceptionV3 載入完成。")

    logger.info("開始訓練迴圈...")
    best_val_metric = float('inf') # 使用 MSE 作為最佳模型判斷標準
    metrics_hist = {'train_loss':[], 'val_loss':[], 'val_mse':[], 'val_mae':[], 'val_mape':[], 'val_smape':[], 'val_fid':[]} # 新增 MAPE, SMAPE

    for epoch in range(1, CONFIG["epochs"] + 1):
        ddpm.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [訓練]", leave=False)
        for x_start, hour_s, day_s, _ in train_pbar: # _ 是 extra_data_rows (目前未使用於損失計算)
            optimizer.zero_grad()
            x_start = x_start.to(CONFIG["device"])
            # hour_s, day_s 是 (N,)，已在 CPU 上，DDPM 內部方法會移至 device
            t = torch.randint(0, CONFIG["timesteps"], (x_start.shape[0],), device=CONFIG["device"]).long() # 隨機取樣時間步 t
            loss = ddpm.p_losses(x_start, t, hour_s, day_s) # 傳遞純量小時/星期
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix({"損失": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        metrics_hist['train_loss'].append(avg_train_loss)

        logger.info(f"Epoch {epoch} - 驗證中...")
        val_metrics = evaluate_model(ddpm, val_loader, inception_fid, CONFIG, CONFIG["fid_num_samples"]//2) # FID 樣本數可調整

        metrics_hist['val_loss'].append(val_metrics['mse']) # val_loss 以 mse 記錄
        metrics_hist['val_mse'].append(val_metrics['mse'])
        metrics_hist['val_mae'].append(val_metrics['mae'])
        metrics_hist['val_mape'].append(val_metrics['mape'])
        metrics_hist['val_smape'].append(val_metrics['smape'])
        metrics_hist['val_fid'].append(val_metrics['fid'])
        logger.info(f"E{epoch}: TrainL:{avg_train_loss:.5f}|ValMSE:{val_metrics['mse']:.5f}|ValMAE:{val_metrics['mae']:.5f}|ValMAPE:{val_metrics['mape']:.2f}%|ValSMAPE:{val_metrics['smape']:.2f}%|ValFID:{val_metrics['fid']:.3f}")

        if val_metrics['mse'] < best_val_metric:
            best_val_metric = val_metrics['mse']
            save_path = os.path.join(CONFIG["save_dir"], "best_ddpm_model.pth")
            torch.save({
                'epoch': epoch,
                'ddpm_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'config': CONFIG, # 儲存當時的組態
                'norm_stats_flow': train_dataset.norm_stats_flow, # 儲存流量標準化統計量
                'sorted_flow_columns': train_dataset.sorted_flow_columns,
                'grid_idx_to_rc_map': train_dataset.grid_idx_to_rc_map,
                'processed_extra_columns': train_dataset.processed_extra_columns,
            }, save_path)
            logger.info(f"已儲存新的最佳模型至 {save_path} (Val MSE: {best_val_metric:.5f})")

        # 每隔一定 epoch 或最後一個 epoch 儲存視覺化樣本
        if epoch % (CONFIG["epochs"]//5 if CONFIG["epochs"] >=5 else 1) == 0 or epoch == CONFIG["epochs"]:
            ddpm.eval() # 確保模型在評估模式
            with torch.no_grad():
                # 從驗證集取一小批資料作視覺化
                fixed_x_s, fixed_hr_s, fixed_day_s, _ = next(iter(val_loader))
                num_viz = min(4, fixed_hr_s.shape[0]) # 最多顯示 4 個樣本
                fixed_hr_s = fixed_hr_s[:num_viz]
                fixed_day_s = fixed_day_s[:num_viz]
                fixed_x_s = fixed_x_s[:num_viz] # 也限制目標樣本數量

                logger.info(f"為 epoch {epoch} 生成視覺化樣本...")
                gen_samples = ddpm.sample(num_viz, fixed_hr_s, fixed_day_s) # (num_viz, C, D, H, W)

                # 反正規化以供視覺化
                mean_v, std_v = train_dataset.norm_stats_flow['mean'], train_dataset.norm_stats_flow['std']
                gen_denorm = gen_samples.cpu() * std_v + mean_v
                orig_denorm = fixed_x_s.cpu() * std_v + mean_v # 目標也需反正規化

                fig, axes = plt.subplots(2, num_viz, figsize=(num_viz*3.5, 7), squeeze=False) # 調整 figsize
                for i in range(num_viz):
                    ax_orig, ax_gen = axes[0,i], axes[1,i]
                    hr_title, dow_title = int(fixed_hr_s[i].item()), int(fixed_day_s[i].item())

                    # 假設 D=1, C=1, 直接 squeeze()
                    im_o = ax_orig.imshow(orig_denorm[i].squeeze().numpy(), cmap='viridis')
                    ax_orig.set_title(f"目標 (H{hr_title} D{dow_title})")
                    ax_orig.axis('off')
                    fig.colorbar(im_o, ax=ax_orig, fraction=0.046, pad=0.04)

                    im_g = ax_gen.imshow(gen_denorm[i].squeeze().numpy(), cmap='viridis')
                    ax_gen.set_title(f"生成 (H{hr_title} D{dow_title})")
                    ax_gen.axis('off')
                    fig.colorbar(im_g, ax=ax_gen, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(os.path.join(CONFIG["save_dir"],f"epoch_{epoch:03d}_samples.png"))
                plt.close(fig)
                logger.info(f"已儲存 epoch {epoch} 的視覺化樣本。")
    logger.info("訓練完成。")

    logger.info("載入最佳模型以進行最終測試集評估...")
    chkpt_path = os.path.join(CONFIG["save_dir"], "best_ddpm_model.pth")
    if not os.path.exists(chkpt_path):
        logger.error(f"找不到最佳模型檔案: {chkpt_path}。跳過最終評估。")
    else:
        chkpt = torch.load(chkpt_path, map_location=CONFIG["device"])
        cfg_chkpt = chkpt.get('config', CONFIG) # 若 checkpoint 無 config，使用當前 CONFIG (向下相容)

        # 使用 checkpoint 中的組態重新初始化模型結構
        final_unet = UNet3D(cfg_chkpt["image_channels"],cfg_chkpt["base_channels_unet"],cfg_chkpt["time_emb_dim"],cfg_chkpt["condition_encode_dim"]).to(CONFIG["device"])
        final_ddpm = DDPM3D(final_unet,cfg_chkpt["timesteps"],(cfg_chkpt["D"],cfg_chkpt["H"],cfg_chkpt["W"]),cfg_chkpt["image_channels"],
                            cfg_chkpt["condition_input_channels"],cfg_chkpt["condition_encode_dim"],
                            beta_start=cfg_chkpt.get("beta_start", CONFIG["beta_start"]), # 向下相容
                            beta_end=cfg_chkpt.get("beta_end", CONFIG["beta_end"]),       # 向下相容
                            device=CONFIG["device"]) # 使用當前執行的 device
        final_ddpm.load_state_dict(chkpt['ddpm_state_dict'])
        logger.info("最佳模型載入完成。")

        test_metrics = evaluate_model(final_ddpm, test_loader, inception_fid, CONFIG, CONFIG["fid_num_samples"]) # 在完整測試集上評估
        logger.info(f"最終測試: MSE:{test_metrics['mse']:.5f}|MAE:{test_metrics['mae']:.5f}|MAPE:{test_metrics['mape']:.2f}%|SMAPE:{test_metrics['smape']:.2f}%|FID:{test_metrics['fid']:.3f}")
        with open(os.path.join(CONFIG["save_dir"], "final_test_metrics.json"),'w') as f: json.dump(test_metrics,f,indent=4)
        with open(os.path.join(CONFIG["save_dir"], "final_test_metrics.txt"),'w') as f:
            f.write(f"最終測試指標 (FID on {CONFIG['fid_num_samples']} samples):\n日期: {pd.Timestamp.now(tz='Asia/Taipei')}\n")
            for k,v in test_metrics.items(): f.write(f"{k.upper()}: {v:.6f}\n")

    try:
        ep_rng = range(1, len(metrics_hist['train_loss']) + 1)
        plt.figure(figsize=(20, 10)); plt.style.use('seaborn_v0_8_darkgrid') # 調整圖片大小

        plt.subplot(2,3,1); plt.plot(ep_rng,metrics_hist['train_loss'],label='訓練損失'); plt.plot(ep_rng,metrics_hist['val_mse'],label='驗證 MSE'); plt.legend(); plt.title('損失/MSE'); plt.grid(True); plt.xlabel("Epoch")
        plt.subplot(2,3,2); plt.plot(ep_rng,metrics_hist['val_mae'],label='驗證 MAE',color='orange'); plt.legend(); plt.title('驗證 MAE'); plt.grid(True); plt.xlabel("Epoch")
        plt.subplot(2,3,3); plt.plot(ep_rng,metrics_hist['val_fid'],label='驗證 FID',color='green'); plt.legend(); plt.title('驗證 FID'); plt.grid(True); plt.xlabel("Epoch")
        plt.subplot(2,3,4); plt.plot(ep_rng,metrics_hist['val_mape'],label='驗證 MAPE (%)',color='purple'); plt.legend(); plt.title('驗證 MAPE'); plt.grid(True); plt.xlabel("Epoch")
        plt.subplot(2,3,5); plt.plot(ep_rng,metrics_hist['val_smape'],label='驗證 SMAPE (%)',color='brown'); plt.legend(); plt.title('驗證 SMAPE'); plt.grid(True); plt.xlabel("Epoch")

        plt.tight_layout(); plt.savefig(os.path.join(CONFIG["save_dir"],"training_history_plots.png")); plt.close()
        logger.info("已儲存訓練歷史圖表。")
    except Exception as e:
        logger.error(f"繪製歷史圖表錯誤: {e}")

    logger.info("================ 腳本執行完成 ================")
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

    # --- 模型參數 ---
    "image_channels": 1,      # 主要資料(流量圖)的通道數
    "condition_input_channels": 2, # 條件處理器接收的原始條件通道數 (小時網格 + 是否假日網格)
    "condition_encode_dim": 16, # 條件處理器輸出的特徵維度 (可調)
    "base_channels_unet": 64,   # UNet3D 的基礎通道數
    "unet_dropout_rate": 0.1,
    "time_emb_dim": 256,        # 時間嵌入維度

    # --- DDPM 參數 ---
    "timesteps": 1000,          # 擴散時間步長
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # --- 訓練參數 ---
    "epochs": 128, # 可調整
    "batch_size": 256, # 依 GPU 記憶體調整
    "lr": 1e-3, # 學習率
    "num_workers": 0, # DataLoader 工作執行緒 (Windows 建議 0, Linux 可 >0)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42, # 隨機種子
    "weight_decay": 1e-5, # 優化器的權重衰減 
    "lr_scheduler_factor": 0.5, # ReduceLROnPlateau: 學習率降低因子
    "lr_scheduler_patience": 4,   # ReduceLROnPlateau: 多少個 epoch 驗證損失未改善則降低學習率
    "lr_scheduler_min_lr": 1e-7,  # ReduceLROnPlateau: 學習率下限
    "early_stopping_patience": 8, # 早停: 多少個 epoch 驗證損失未改善則停止訓練 
    "resume_from_checkpoint": True,  # 是否嘗試從檢查點恢復訓練
    "checkpoint_path": "best_ddpm_model_training.pth", #預設使用的檢查點檔案名

    # --- 評估參數 ---
    "eval_batch_size": 256,
    "fid_batch_size": 256,
    "fid_num_samples": 128, # FID 計算樣本數

    # --- 路徑與儲存 ---
    "save_dir": "results_ddpm_long-term", # 結果儲存目錄
    "plot_grid_mapping_path": "grid_mapping_visualization_taipei.png", # 網格映射視覺化圖片路徑
    "train_split_ratio": 0.7, # 訓練集比例
    "val_split_ratio": 0.15,  # 驗證集比例
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
logger.info(f"結果將儲存於: {CONFIG['save_dir']}")

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])
logger.info(f"使用裝置: {CONFIG['device']}")

# --------------------------------------
# 數據處理相關
# --------------------------------------
def parse_lat_lon(column_name: str) -> tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class PeopleFlowDatasetCondition(Dataset):
    def __init__(self,
                 df: pd.DataFrame, # 傳入 DataFrame 物件
                 config: Dict[str, Any],
                 mode: str = 'train',
                 # 驗證/測試模式下，由訓練資料集實例傳入
                 average_flow_map_dict: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                 norm_stats_flow: Optional[Dict[str, float]] = None,
                 sorted_flow_columns_from_train: Optional[List[str]] = None,
                 grid_idx_to_rc_map_from_train: Optional[Dict[int, Tuple[int,int]]] = None,
                 processed_extra_columns_from_train: Optional[List[str]] = None,
                 selected_sensor_info_from_train: Optional[List[Dict[str, Any]]] = None
                ):
        super().__init__()
        self.df_original = df.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.H = config["H"]
        self.W = config["W"]
        self.D = config.get("D", 1)
        self.image_channels = config.get("image_channels", 1)
        self.num_grid_cells = self.H * self.W
        logger = logging.getLogger(__name__) # 確保 logger 可用

        # --- 時間解析 ---
        actual_datetime_col = '時間'
        if actual_datetime_col not in self.df_original.columns:
            raise ValueError(f"資料中未找到指定的日期時間欄位 '{actual_datetime_col}'。")
        df_datetime_processed = self.df_original.copy()
        df_datetime_processed[actual_datetime_col] = pd.to_datetime(df_datetime_processed[actual_datetime_col])
        
        self.hours_original_np = df_datetime_processed[actual_datetime_col].dt.hour.values
        self.hour_category_for_grouping_np = (self.hours_original_np > 8).astype(int)
        logger.info(f"已生成用於分組的小時類別 (0: hr <= 8, 1: hr > 8)。 "
                            f"類別0 數量: {np.sum(self.hour_category_for_grouping_np == 0)}, "
                            f"類別1 數量: {np.sum(self.hour_category_for_grouping_np == 1)}")

        # --- 新增：明確儲存假日狀態 ---
        holiday_col_name = 'holiday' # <--- 確認你的假日欄位名稱
        if holiday_col_name not in self.df_original.columns:
             if "hoilday" in self.df_original.columns: # 嘗試修正可能的拼寫錯誤
                 logger.warning(f"找到 'hoilday' 欄位，將其更名為 'holiday'。")
                 self.df_original.rename(columns={"hoilday": "holiday"}, inplace=True)
                 df_datetime_processed.rename(columns={"hoilday": "holiday"}, inplace=True) # 同步修改複製的df
             else:
                 raise ValueError(f"資料中未找到指定的假日欄位 '{holiday_col_name}'。")

        # 確保假日欄位是數值 0 或 1
        if self.df_original[holiday_col_name].dtype == bool:
            self.is_holiday_original_np = self.df_original[holiday_col_name].astype(int).values
        elif pd.api.types.is_numeric_dtype(self.df_original[holiday_col_name]):
             unique_holiday_vals = self.df_original[holiday_col_name].dropna().unique()
             if not all(np.isclose(v, 0) or np.isclose(v, 1) for v in unique_holiday_vals if not np.isnan(v)): # 排除NaN再比較
                 logger.warning(f"假日欄位 '{holiday_col_name}' 包含非 0 或 1 的數值: {unique_holiday_vals[:10]}... "
                                f"將把非0值視為1(假日)，0值視為0(非假日)。請確認此邏輯。")
             # 將非0值視為1 (假日)，0值視為0 (非假日)，缺失值填0
             self.is_holiday_original_np = self.df_original[holiday_col_name].fillna(0).astype(bool).astype(int).values
        else: # 其他類型，嘗試轉換
            try:
                # 先嘗試將常見的文字表示轉為0/1
                if self.df_original[holiday_col_name].dtype == object:
                    # 假設 '是', 'True', '1', 'Y', 'Yes' 代表假日
                    holiday_map = {
                        '是': 1, 'true': 1, '1': 1, 'yes': 1, 'y': 1,
                        '否': 0, 'false': 0, '0': 0, 'no': 0, 'n': 0
                    }
                    # 先轉小寫處理
                    temp_series = self.df_original[holiday_col_name].astype(str).str.lower().map(holiday_map)
                else:
                    temp_series = pd.Series(np.nan, index=self.df_original.index) # 初始化為 NaN

                # 對於未能通過 map 轉換的值 (即仍為 NaN 的)，再嘗試 pd.to_numeric
                numeric_conversion_needed_mask = temp_series.isna()
                if numeric_conversion_needed_mask.any():
                    numeric_converted_part = pd.to_numeric(self.df_original.loc[numeric_conversion_needed_mask, holiday_col_name], errors='coerce')
                    temp_series.loc[numeric_conversion_needed_mask] = numeric_converted_part
                
                num_failed_conversions = temp_series.isna().sum() - self.df_original[holiday_col_name].isna().sum() # 計算新產生的NaN
                self.is_holiday_original_np = temp_series.fillna(0).astype(bool).astype(int).values # 最終轉換為 0/1
                
                if num_failed_conversions > 0:
                    logger.warning(f"假日欄位 '{holiday_col_name}' 包含無法直接解析為0/1的值，已嘗試轉換，其中 {num_failed_conversions} 個值被視為 0 (非假日)。")
            except Exception as e:
                raise TypeError(f"無法處理假日欄位 '{holiday_col_name}' 的資料型態。請確保它是數值 (0/1)、布林值或可明確轉換的文字。錯誤: {e}")
        logger.info(f"已處理 'is_holiday_original_np'，其中假日 (1) 的數量: {np.sum(self.is_holiday_original_np)}, "
                    f"非假日 (0) 的數量: {len(self.is_holiday_original_np) - np.sum(self.is_holiday_original_np)}")


        # --- 處理額外資料 (這部分通常不變) ---
        df_for_extra_processing = self.df_original.copy()
        if actual_datetime_col in df_for_extra_processing.columns:
            temp_dt_series = pd.to_datetime(df_for_extra_processing[actual_datetime_col], errors='coerce')
            if temp_dt_series.notna().all(): # 確保轉換成功
                df_for_extra_processing['年'] = temp_dt_series.dt.year
                df_for_extra_processing['月'] = temp_dt_series.dt.month
                df_for_extra_processing['日'] = temp_dt_series.dt.day
                df_for_extra_processing['時'] = temp_dt_series.dt.hour 
                df_for_extra_processing['weekday'] = temp_dt_series.dt.dayofweek
            else:
                logger.error(f"df_for_extra_processing 中時間欄位 '{actual_datetime_col}' 包含無法解析的日期，額外時間特徵可能不正確。")
        else:
            logger.warning(f"df_for_extra_processing 中未找到時間欄位 '{actual_datetime_col}'，部分額外時間特徵可能無法生成。")

        self.extra_cols_list_definition = config.get("extra_features_definition_list", [
            "測站氣壓", "海平面氣壓", "氣溫", "露點溫度", "相對溼度", "風速", "最大陣風",
            "降水量", "降水時數", "日照時數", "全天空日射量", "能見度", "紫外線指數", "總雲量",
            "holiday", "weekday", "年", "月", "日", "時" 
        ])
        wind_cols_to_process = []
        if '風向' in df_for_extra_processing.columns: wind_cols_to_process.append('風向')
        if '最大陣風風向' in df_for_extra_processing.columns: wind_cols_to_process.append('最大陣風風向')
        for col in wind_cols_to_process:
            # 嘗試轉換風向欄位為數值，處理可能的錯誤
            try:
                df_for_extra_processing[col] = pd.to_numeric(df_for_extra_processing[col], errors='coerce').fillna(0) # 填充無法轉換的為0
                df_for_extra_processing[f'sin_{col}'] = np.sin(np.deg2rad(df_for_extra_processing[col].astype(float)))
                df_for_extra_processing[f'cos_{col}'] = np.cos(np.deg2rad(df_for_extra_processing[col].astype(float)))
                if f'sin_{col}' not in self.extra_cols_list_definition: self.extra_cols_list_definition.append(f'sin_{col}')
                if f'cos_{col}' not in self.extra_cols_list_definition: self.extra_cols_list_definition.append(f'cos_{col}')
            except Exception as e:
                logger.warning(f"處理風向欄位 {col} 時出錯: {e}. 將跳過此欄位的 sin/cos 轉換。")
        self.extra_cols_list_definition = [col for col in self.extra_cols_list_definition if col not in ['風向', '最大陣風風向']]
        
        present_extra_cols = [col for col in self.extra_cols_list_definition if col in df_for_extra_processing.columns]
        df_extra_subset = df_for_extra_processing[present_extra_cols].copy()
        
        if "hoilday" in df_extra_subset.columns and "holiday" not in df_extra_subset.columns:
            df_extra_subset.rename(columns={"hoilday": "holiday"}, inplace=True)

        cat_features = []
        if 'holiday' in present_extra_cols: cat_features.append('holiday')
        if 'weekday' in present_extra_cols: cat_features.append('weekday')
        
        actual_cat_features = [col for col in cat_features if col in df_extra_subset.columns]
        if actual_cat_features:
            df_extra_subset[actual_cat_features] = df_extra_subset[actual_cat_features].astype(str)
            df_cat = pd.get_dummies(df_extra_subset[actual_cat_features], prefix=actual_cat_features, dummy_na=False)
        else:
            df_cat = pd.DataFrame(index=df_extra_subset.index) # 確保索引對齊
        
        df_cont = df_extra_subset.drop(columns=actual_cat_features, errors='ignore')
        # 確保 df_cont 中的所有欄位都是數值型態，並填充 NaN
        for col in df_cont.columns:
            if not pd.api.types.is_numeric_dtype(df_cont[col]):
                df_cont[col] = pd.to_numeric(df_cont[col], errors='coerce')
        df_cont = df_cont.fillna(0)

        df_extra_processed = pd.concat([df_cont, df_cat], axis=1)

        if self.mode == 'train':
            self.processed_extra_columns = list(df_extra_processed.columns)
            self.processed_extra_data_np = df_extra_processed.values.astype(np.float32) # fillna(0) 已在上一步驟對 df_cont 完成
            logger.info(f"訓練集: 已處理 {len(self.processed_extra_columns)} 個額外特徵。")
        else:
            if processed_extra_columns_from_train is None:
                raise ValueError("驗證/測試模式，必須提供 processed_extra_columns_from_train。")
            self.processed_extra_columns = processed_extra_columns_from_train
            df_extra_processed = df_extra_processed.reindex(columns=self.processed_extra_columns, fill_value=0)
            self.processed_extra_data_np = df_extra_processed.values.astype(np.float32) # fillna(0) 已在上一步驟對 df_cont 完成，reindex 會用 fill_value=0 處理新欄位
            logger.info(f"{self.mode} 資料集: 已處理額外特徵。")
            
        if self.mode == 'train':
            all_flow_columns_with_coords = [c for c in self.df_original.columns if '(' in c and ')' in c]
            logger.info(f"從欄位名稱中找到 {len(all_flow_columns_with_coords)} 個可能的流量/座標欄位。")
            num_required_points = self.H * self.W
            if len(all_flow_columns_with_coords) < num_required_points:
                raise ValueError(
                    f"網格大小 ({self.H}x{self.W}={num_required_points}) 大於了可用的地理座標點數量 ({len(all_flow_columns_with_coords)}). "
                    "請減少 H*W 或提供更多座標點。"
                )
            all_column_info_list = []
            for col_name in all_flow_columns_with_coords:
                try:
                    lon, lat = parse_lat_lon(col_name)
                    all_column_info_list.append({'name': col_name, 'lon': lon, 'lat': lat})
                except ValueError as e:
                    logger.warning(f"無法解析欄位 '{col_name}' 的座標: {e}。將跳過此欄位。")

            if len(all_column_info_list) < num_required_points:
                 raise ValueError(
                    f"成功解析的地理座標點數量 ({len(all_column_info_list)}) 不足所需的網格點 ({num_required_points})。請檢查資料欄位格式或減少 H*W。"
                )
            all_coords_np = np.array([(info['lon'], info['lat']) for info in all_column_info_list])
            self.selected_sensor_info = []
            selected_real_coords_np_for_grid_def = None
            if len(all_column_info_list) > num_required_points:
                logger.info(f"座標點數量 ({len(all_column_info_list)}) 多於網格數 ({num_required_points}). "
                             f"將選擇最靠近地理中心的 {num_required_points} 個座標點進行映射。")
                geometric_center_lon = np.mean(all_coords_np[:, 0])
                geometric_center_lat = np.mean(all_coords_np[:, 1])
                distances_to_geometric_center = np.sqrt(
                    (all_coords_np[:, 0] - geometric_center_lon)**2 +
                    (all_coords_np[:, 1] - geometric_center_lat)**2
                )
                selected_indices = np.argsort(distances_to_geometric_center)[:num_required_points]
                self.selected_sensor_info = [all_column_info_list[i] for i in selected_indices]
                selected_real_coords_np_for_grid_def = all_coords_np[selected_indices]
            else:
                self.selected_sensor_info = all_column_info_list
                selected_real_coords_np_for_grid_def = all_coords_np
            logger.info(f"已選定 {len(self.selected_sensor_info)} 個感測器進行網格映射。")

            if selected_real_coords_np_for_grid_def is None or selected_real_coords_np_for_grid_def.shape[0] != num_required_points:
                 raise ValueError(f"用於定義網格的選定座標點數量 ({selected_real_coords_np_for_grid_def.shape[0] if selected_real_coords_np_for_grid_def is not None else 0}) 與所需網格點 ({num_required_points}) 不符。")

            self.grid_target_coords, self.grid_idx_to_rc_map = self._define_target_grid_cells_hierarchical_style(selected_real_coords_np_for_grid_def)
            self.sorted_flow_columns = self._map_sensors_to_target_grid_hungarian(
                self.selected_sensor_info,
                selected_real_coords_np_for_grid_def,
                self.grid_target_coords
            )
            plot_path = os.path.join(self.config["save_dir"], self.config.get("plot_grid_mapping_path", "grid_mapping_visualization.png"))
            self._plot_grid_mapping(
                self.grid_idx_to_rc_map,
                self.sorted_flow_columns,
                plot_path
            )
            self.average_flow_map_dict = self._calculate_average_flows()

            all_avg_flows_list = [flow for flow in self.average_flow_map_dict.values() if flow is not None]
            if not all_avg_flows_list:
                raise ValueError("訓練集中未計算出任何平均流量。無法計算流量標準化統計量。")
            all_avg_flows_np = np.stack(all_avg_flows_list)
            self.flow_mean_val = np.mean(all_avg_flows_np)
            self.flow_std_val = np.std(all_avg_flows_np)
            if self.flow_std_val < 1e-5: self.flow_std_val = 1e-5
            self.norm_stats_flow = {'mean': self.flow_mean_val, 'std': self.flow_std_val}
            logger.info(f"訓練集流量標準化統計量: 平均值={self.flow_mean_val:.4f}, 標準差={self.flow_std_val:.4f}")
        else:
            if not all([average_flow_map_dict, norm_stats_flow, sorted_flow_columns_from_train,
                        grid_idx_to_rc_map_from_train, processed_extra_columns_from_train,
                        selected_sensor_info_from_train]):
                raise ValueError("驗證/測試模式下，必須提供所有必要的預計算資料。")
            self.average_flow_map_dict = average_flow_map_dict
            self.norm_stats_flow = norm_stats_flow
            self.flow_mean_val = self.norm_stats_flow['mean']
            self.flow_std_val = self.norm_stats_flow['std']
            if self.flow_std_val < 1e-5: self.flow_std_val = 1e-5
            self.sorted_flow_columns = sorted_flow_columns_from_train
            self.grid_idx_to_rc_map = grid_idx_to_rc_map_from_train
            self.processed_extra_columns = processed_extra_columns_from_train
            self.selected_sensor_info = selected_sensor_info_from_train
            logger.info(f"{self.mode} 資料集使用預計算的流量標準化統計量、欄位順序、網格映射和感測器資訊。")

    def _define_target_grid_cells_hierarchical_style(self, selected_real_coords_np: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
        logger = logging.getLogger(__name__)
        if selected_real_coords_np.shape[0] != self.num_grid_cells:
            raise ValueError(f"selected_real_coords_np 應含 {self.num_grid_cells} 點, 得到 {selected_real_coords_np.shape[0]}。")
        logger.info("使用 'hierarchical' 風格定義目標網格中心點...")
        center_lon, center_lat = np.mean(selected_real_coords_np, axis=0)
        unique_lons = np.unique(selected_real_coords_np[:,0])
        unique_lats = np.unique(selected_real_coords_np[:,1])
        lon_diffs = np.diff(np.sort(unique_lons))
        lat_diffs = np.diff(np.sort(unique_lats))
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

    def _map_sensors_to_target_grid_hungarian(self,
                                            sel_sensor_info_list: List[Dict[str,Any]],
                                            sel_coords_np: np.ndarray,
                                            grid_target_coords_np: np.ndarray
                                            ) -> List[str]:
        logger = logging.getLogger(__name__)
        logger.info("使用匈牙利演算法將選定感測器映射到目標網格...")
        n_sensors = sel_coords_np.shape[0]
        n_targets = grid_target_coords_np.shape[0]
        if n_sensors != self.num_grid_cells or n_targets != self.num_grid_cells:
            raise ValueError(f"感測器數量 ({n_sensors}) 或目標網格點數量 ({n_targets}) "
                             f"必須等於 H*W ({self.num_grid_cells})。")
        if len(sel_sensor_info_list) != n_sensors:
             raise ValueError(f"sel_sensor_info_list 的長度 ({len(sel_sensor_info_list)}) 與 sel_coords_np ({n_sensors}) 不符。")
        costs = np.sqrt(np.sum((sel_coords_np[:, np.newaxis, :] - grid_target_coords_np[np.newaxis, :, :])**2, axis=2))
        assigned_real_indices, assigned_target_indices = linear_sum_assignment(costs)
        target_to_real_map = {t_idx: r_idx for r_idx, t_idx in zip(assigned_real_indices, assigned_target_indices)}
        sorted_flow_column_names = [""] * self.num_grid_cells
        for flat_target_idx in range(self.num_grid_cells):
            if flat_target_idx in target_to_real_map:
                real_idx_in_sel_list = target_to_real_map[flat_target_idx]
                if real_idx_in_sel_list < len(sel_sensor_info_list):
                    sorted_flow_column_names[flat_target_idx] = sel_sensor_info_list[real_idx_in_sel_list]['name']
                else:
                     raise IndexError(f"匈牙利演算法映射的真實索引 {real_idx_in_sel_list} 超出 sel_sensor_info_list 的範圍 (長度 {len(sel_sensor_info_list)})。")
            else:
                 raise ValueError(f"目標網格索引 {flat_target_idx} 未在匈牙利演算法的映射結果中找到。成本矩陣可能存在問題。")
        logger.info(f"成功為網格排序 {len(sorted_flow_column_names)} 個流量欄位。")
        return sorted_flow_column_names
        
    def _calculate_average_flows(self) -> Dict[Tuple[int, int], np.ndarray]:
        """計算每個 (小時, 是否假日) 組合的平均流量圖。"""
        logger = logging.getLogger(__name__)
        logger.info("計算 (小時類別, 是否為假日) 平均流量圖...") 
        avg_flows = {}
        if not hasattr(self, 'sorted_flow_columns') or not self.sorted_flow_columns or any(col == "" for col in self.sorted_flow_columns):
            raise AttributeError("Dataset object missing 'sorted_flow_columns' or it's invalid. Cannot calculate average flows.")
        
        missing_cols = [col for col in self.sorted_flow_columns if col not in self.df_original.columns]
        if missing_cols:
            raise ValueError(f"以下流量欄位在 DataFrame 中未找到: {missing_cols}")

        flow_data_grid_alltimes = self.df_original[self.sorted_flow_columns].values.astype(np.float32)
        
        # 使用 self.hours_original_np 和 self.is_holiday_original_np 進行分組
        grouping_df = pd.DataFrame({
            'hour_category': self.hour_category_for_grouping_np,
            'is_holiday': self.is_holiday_original_np # <--- 修改分組依據
        })
        grouped = grouping_df.groupby(['hour_category', 'is_holiday'])

        if not grouped.groups:
            logger.warning("無法根據 (小時, 是否為假日) 對資料進行分組。")
            return {} 

        for (hr_cat, is_hol), group_indices in grouped.groups.items(): # 新
            if len(group_indices) == 0:
                logger.warning(f"條件 (hour_category={hr_cat}, is_holiday={is_hol}) 沒有對應的資料。") # 新
                continue

            group_flows_for_condition = flow_data_grid_alltimes[group_indices]
            mean_flow_flat_for_condition = np.nanmean(group_flows_for_condition, axis=0)
            mean_flow_flat_for_condition[np.isnan(mean_flow_flat_for_condition)] = 0 
            avg_flows[(hr_cat, int(is_hol))] = mean_flow_flat_for_condition.reshape(self.H, self.W)

        if not avg_flows:
            logger.warning("未計算任何 (小時類別, 是否為假日) 的平均流量。")
        logger.info(f"計算完成 {len(avg_flows)} 個 (小時類別, 是否為假日) 條件的平均流量圖。") # <--- 修改日誌訊息
        return avg_flows

    def _plot_grid_mapping(self, grid_idx_to_rc_map: Dict[int, Tuple[int,int]], sorted_flow_cols: List[str], save_path: str):
        logger = logging.getLogger(__name__)
        plt.figure(figsize=(12, 12)); plt.style.use('seaborn-v0_8-whitegrid')
        if not hasattr(self, 'selected_sensor_info') or not self.selected_sensor_info or \
           not hasattr(self, 'sorted_flow_columns') or not self.sorted_flow_columns or \
           len(self.sorted_flow_columns) != self.num_grid_cells:
            logger.error("_plot_grid_mapping: 內部屬性未正確初始化或長度不符。")
            plt.close(); return
        selected_sensor_info_dict = {info['name']: (info['lon'], info['lat']) for info in self.selected_sensor_info}
        actual_sensor_lons, actual_sensor_lats, grid_labels = [], [], []
        for flat_idx, col_name in enumerate(sorted_flow_cols):
            if col_name in selected_sensor_info_dict:
                lon, lat = selected_sensor_info_dict[col_name]
                actual_sensor_lons.append(lon); actual_sensor_lats.append(lat)
                r_map, c_map = grid_idx_to_rc_map.get(flat_idx, (-1,-1))
                grid_labels.append(f'[{r_map},{c_map}]' if r_map!=-1 else f'[{flat_idx}]')
            elif col_name and col_name.strip(): # 只有當 col_name 非空且非空白時才警告
                logger.warning(f"欄位 '{col_name}' (索引 {flat_idx}) 在 selected_sensor_info_dict 中未找到。")
        if not actual_sensor_lons: logger.warning("無實際感測器座標點可繪製。"); plt.close(); return
        plt.scatter(actual_sensor_lons, actual_sensor_lats, c='blue', marker='o', s=50, alpha=0.7, label='Grid Points (Actual Sensor Location)')
        for i, lbl in enumerate(grid_labels): plt.text(actual_sensor_lons[i], actual_sensor_lats[i], lbl, fontsize=7, color='navy', ha='right', va='bottom')
        plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title(f"Grid Mapping ({self.H}x{self.W})")
        plt.grid(True, linestyle=':', alpha=0.6); plt.gca().set_aspect('equal', adjustable='box')
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); logger.info(f"Grid mapping visualization saved to {save_path}")
        except Exception as e: logger.error(f"無法儲存網格映射圖至 {save_path}: {e}")
        plt.close()

    def __len__(self) -> int:
        return len(self.df_original)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        logger = logging.getLogger(__name__)
        current_hour_original = self.hours_original_np[idx]
        current_hour_category_for_grouping = self.hour_category_for_grouping_np[idx] # 新增，用於查找目標
        current_is_holiday_original = self.is_holiday_original_np[idx]

        target_avg_flow_np = self.average_flow_map_dict.get((current_hour_category_for_grouping, current_is_holiday_original)) 

        if target_avg_flow_np is None:
            logger.warning(f"在 average_flow_map_dict 中找不到 (hour={current_hour_original}, is_holiday={current_is_holiday_original}) 的平均流量 (索引 {idx})，將使用零值網格。")
            target_avg_flow_np = np.zeros((self.H, self.W), dtype=np.float32)

        if not hasattr(self, 'flow_mean_val') or not hasattr(self, 'flow_std_val'):
            raise AttributeError("flow_mean_val 或 flow_std_val 未在 Dataset 初始化時設定。")

        std_val_safe = self.flow_std_val if self.flow_std_val > 1e-6 else 1.0
        standardized_avg_flow_np = (target_avg_flow_np - self.flow_mean_val) / std_val_safe
        target_flow_tensor = torch.from_numpy(standardized_avg_flow_np).float()

        # Reshape (保持不變)
        if self.D == 1 and self.image_channels == 1:
            target_flow_tensor = target_flow_tensor.unsqueeze(0).unsqueeze(0) 
        elif self.image_channels > 1 or self.D > 1:
            target_flow_tensor = target_flow_tensor.unsqueeze(0)
            if self.D > 1: target_flow_tensor = target_flow_tensor.repeat(self.D, 1, 1)
            if self.image_channels > 1:
                target_flow_tensor = target_flow_tensor.unsqueeze(0).repeat(self.image_channels, 1, 1, 1)
                logger.warning(f"Target flow image_channels > 1. Repeating the single flow channel.")
            else: target_flow_tensor = target_flow_tensor.unsqueeze(0)
            expected_shape = (self.image_channels, self.D, self.H, self.W)
            if target_flow_tensor.shape != expected_shape:
                logger.warning(f"Final target_flow_tensor shape {target_flow_tensor.shape} != expected {expected_shape}.")
        else: target_flow_tensor = target_flow_tensor.unsqueeze(0).unsqueeze(0)

        if not hasattr(self, 'processed_extra_data_np'):
            raise AttributeError("processed_extra_data_np 未在 Dataset 初始化時設定。")
        extra_data_row_tensor = torch.from_numpy(self.processed_extra_data_np[idx]).float()

        # --- 返回修改後的元組: (流量, 小時, 是否假日, 額外特徵) ---
        return target_flow_tensor, int(current_hour_original), int(current_is_holiday_original), extra_data_row_tensor     

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
    """3D Denoising Diffusion Probabilistic Model (條件: 小時 + 是否假日)"""
    def __init__(self,
                 unet_model: UNet3D, # 假設 UNet3D 類別已定義
                 timesteps: int,
                 image_size: Tuple[int, int, int], # (D, H, W)
                 image_channels: int,
                 condition_input_channels: int, # 條件處理器輸入的原始通道數 (小時網格+假日網格 = 2)
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

        # --- 條件處理器 (接收 2 個通道: 小時網格 + 假日狀態網格) ---
        self.condition_processor = nn.Sequential(
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False), 
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False), 
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()
        ).to(device)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t) 
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        sact = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        soma_ct = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sact * x_start + soma_ct * noise 

    def _prepare_conditional_input_grids(self,
                                    hour_scalars_batch: torch.Tensor,     # (N,) 原始 0-23
                                    is_holiday_scalars_batch: torch.Tensor, # (N,) 原始 0/1 <--- 修改
                                    ) -> torch.Tensor: # 輸出 (N, 2, D, H, W)
        """將純量的小時和假日狀態轉換為正規化的網格輸入""" # <--- 修改註解
        logger = logging.getLogger(__name__) # 添加 logger
        batch_size = hour_scalars_batch.shape[0]
        if hour_scalars_batch.shape[0] != is_holiday_scalars_batch.shape[0]:
            logger.error(f"Batch size mismatch in _prepare_conditional_input_grids: hour_batch={hour_scalars_batch.shape[0]}, holiday_batch={is_holiday_scalars_batch.shape[0]}")
            raise ValueError("Batch sizes for hour and holiday scalars must match.")

        # 小時正規化
        norm_hours = hour_scalars_batch.float().to(self.device) / 23.0 
        
        # 假日狀態 (假設已是 0 或 1，不需要額外正規化)
        holiday_values = is_holiday_scalars_batch.float().to(self.device) 

        # 建立網格
        hour_grid_vals = norm_hours.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)
        holiday_grid_vals = holiday_values.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W) # <--- 使用 holiday_values

        hour_grids_t = hour_grid_vals.unsqueeze(1).unsqueeze(2) 
        holiday_grids_t = holiday_grid_vals.unsqueeze(1).unsqueeze(2) # <--- 修改變數名

        if self.image_size_D != 1:
            hour_grids_t = hour_grids_t.repeat(1,1,self.image_size_D,1,1)
            holiday_grids_t = holiday_grids_t.repeat(1,1,self.image_size_D,1,1) # <--- 修改變數名

        return torch.cat((hour_grids_t, holiday_grids_t), dim=1)

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor,
             hour_scalars_batch: torch.Tensor, 
             is_holiday_scalars_batch: torch.Tensor, # <--- 修改參數名
             noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise) 
        stacked_cond_grids = self._prepare_conditional_input_grids(hour_scalars_batch, is_holiday_scalars_batch) # <--- 修改傳遞的變數
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
        if t_scalar == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_tensor_batch, x_t.shape)
            noise = torch.randn_like(x_t) 
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int,...], 
                  hour_scalars_batch: torch.Tensor, 
                  is_holiday_scalars_batch: torch.Tensor) -> torch.Tensor: # <--- 修改參數名
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device) 
        stacked_cond_grids = self._prepare_conditional_input_grids(hour_scalars_batch, is_holiday_scalars_batch) # <--- 修改傳遞的變數
        processed_conditions = self.condition_processor(stacked_cond_grids) 
        for i in tqdm(reversed(range(0, self.timesteps)), desc="DDPM Sampling Loop", total=self.timesteps, leave=False):
            t_tensor_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, i, t_tensor_batch, processed_conditions)
        return img

    @torch.no_grad()
    def sample(self, batch_size: int, 
           hour_scalars_batch: torch.Tensor, 
           is_holiday_scalars_batch: torch.Tensor) -> torch.Tensor: # <--- 修改參數名
        if hour_scalars_batch.shape[0] != batch_size or is_holiday_scalars_batch.shape[0] != batch_size:
            raise ValueError(f"Provided hour/holiday scalars batch size ({hour_scalars_batch.shape[0]}/{is_holiday_scalars_batch.shape[0]}) "
                            f"does not match requested batch_size ({batch_size})")
        s = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        return self.p_sample_loop(s, hour_scalars_batch, is_holiday_scalars_batch)

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

def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    # (與 DDPM_3DUNet.ipynb 中的定義相同)
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

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
                        dataset_for_coords: Any, # 實際應為 PeopleFlowDatasetCondition 實例
                        error_metrics_grids: Dict[str, np.ndarray], # 例如: {'MSE': mse_grid, 'MAE': mae_grid, ...}
                        config: Dict[str, Any],
                        prefix: str = "test_eval"
                       ):
    """
    在地理座標上繪製每個網格點的平均誤差。
    標籤和標題已修改為英文。
    色彩映射修改為紅到黑，圖內數字為白色整數（MSE圖不顯示數字）。
    """
    logger = logging.getLogger(__name__) # 確保 logger 在函數作用域內可用
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    H, W = config["H"], config["W"]
    if not hasattr(dataset_for_coords, 'sorted_flow_columns') or \
       not hasattr(dataset_for_coords, 'grid_idx_to_rc_map') or \
       not hasattr(dataset_for_coords, 'selected_sensor_info'):
        logger.error("Dataset instance lacks necessary grid mapping information (sorted_flow_columns, grid_idx_to_rc_map, selected_sensor_info).")
        return

    selected_sensor_info_dict = {info['name']: (info['lon'], info['lat']) for info in dataset_for_coords.selected_sensor_info}

    actual_sensor_lons = []
    actual_sensor_lats = []
    valid_grid_indices_flat = []

    for flat_grid_idx in range(H * W):
        if flat_grid_idx < len(dataset_for_coords.sorted_flow_columns):
            col_name = dataset_for_coords.sorted_flow_columns[flat_grid_idx]
            if col_name in selected_sensor_info_dict:
                lon, lat = selected_sensor_info_dict[col_name]
                actual_sensor_lons.append(lon)
                actual_sensor_lats.append(lat)
                valid_grid_indices_flat.append(flat_grid_idx)
            else:
                logger.warning(f"plot_grid_with_error: Column {col_name} (expected at grid index {flat_grid_idx}) not found in selected_sensor_info_dict.")
        else:
            logger.warning(f"plot_grid_with_error: flat_grid_idx {flat_grid_idx} is out of bounds for sorted_flow_columns (length: {len(dataset_for_coords.sorted_flow_columns)}).")

    if not actual_sensor_lons:
        logger.error("plot_grid_with_error: Could not retrieve coordinates for any grid points.")
        return

    # 定義從紅色到黑色的色彩映射
    # 紅色 (低值) -> 黑色 (高值)
    cdict_red_to_black = {
        'red':   ((0.0, 1.0, 1.0),  # 在 0.0 (低值) 時，紅色為 1
                  (1.0, 0.0, 0.0)), # 在 1.0 (高值) 時，紅色為 0
        'green': ((0.0, 0.0, 0.0),  # 綠色始終為 0
                  (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0),  # 藍色始終為 0
                  (1.0, 0.0, 0.0))
    }
    red_to_black_cmap = mcolors.LinearSegmentedColormap('RedToBlack', cdict_red_to_black)

    for metric_name, error_grid_flat in error_metrics_grids.items():
        if error_grid_flat.shape[0] != H*W :
            logger.error(f"Dimension of error_grid for metric {metric_name} ({error_grid_flat.shape}) is incorrect. Expected ({H*W},). Skipping plot.")
            continue

        error_values_for_plot = error_grid_flat[valid_grid_indices_flat]
        
        if len(error_values_for_plot) == 0:
            logger.warning(f"No valid error values to plot for metric {metric_name} after filtering by valid_grid_indices_flat. Skipping plot.")
            continue

        plt.figure(figsize=(12, 12))
        # 使用新的 red_to_black_cmap
        scatter = plt.scatter(actual_sensor_lons, actual_sensor_lats, c=error_values_for_plot, cmap=red_to_black_cmap, marker='s', s=100)
        plt.colorbar(scatter, label=metric_name)

        # 只有非 MSE 的指標圖才在網格上顯示數字
        if metric_name.upper() != 'MSE':
            for i in range(len(actual_sensor_lons)):
                val_to_text = error_values_for_plot[i]
                plt.text(actual_sensor_lons[i], actual_sensor_lats[i],
                         f'{val_to_text:.0f}', # 修改為顯示整數
                         fontsize=6, color='white', ha='center', va='center') # 修改文字顏色為白色

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Geographic Grid Error Heatmap - {metric_name.upper()}")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(save_dir, f'{prefix}_grid_error_map_{metric_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {metric_name} geographic grid error map.")

    # evaluate_model 函數
# (假設其定義與先前完整腳本相同，
# 但需傳遞純量小時/星期至 ddpm_model.sample)
@torch.no_grad()
def evaluate_model(ddpm_model: DDPM3D, 
                   dataloader: torch.utils.data.DataLoader, 
                   inception_model_fid: torch.nn.Module, 
                   config: Dict[str, Any], # 使用 Any 以適應 config 的多樣性
                   max_samples_for_fid: Optional[int] = None
                   ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]: 
    logger = logging.getLogger(__name__)
    ddpm_model.eval()
    inception_model_fid.eval()

    all_generated_samples_for_fid: List[torch.Tensor] = [] # 明確類型
    all_original_samples_for_fid: List[torch.Tensor] = []  # 明確類型
    all_generated_denorm_list: List[torch.Tensor] = [] # 明確類型
    all_original_denorm_list: List[torch.Tensor] = []  # 明確類型
    
    max_fid_samples_limit = len(dataloader.dataset) # type: ignore
    if max_samples_for_fid is not None:
         max_fid_samples_limit = min(max_samples_for_fid, len(dataloader.dataset)) # type: ignore

    dataset_obj = dataloader.dataset # type: ignore
    if not hasattr(dataset_obj, 'norm_stats_flow') or dataset_obj.norm_stats_flow is None: # type: ignore
        raise AttributeError("dataloader.dataset does not have 'norm_stats_flow'.")
    mean_val: float = dataset_obj.norm_stats_flow['mean'] # type: ignore
    std_val: float = dataset_obj.norm_stats_flow['std'] # type: ignore
    if std_val < 1e-6: std_val = 1.0
    
    pbar = tqdm(dataloader, desc="Evaluating Model", leave=False)
    for batch_idx, data_tuple in enumerate(pbar):
        # DataLoader 解包: (target_flow_tensor, hour_original, is_holiday_original, extra_data_row_tensor)
        target_avg_flow_norm, hour_scalars, is_holiday_scalars, _ = data_tuple

        current_batch_size = target_avg_flow_norm.shape[0]
        target_avg_flow_norm = target_avg_flow_norm.to(config["device"])
        hour_scalars = hour_scalars.to(config["device"])
        is_holiday_scalars = is_holiday_scalars.to(config["device"]) 

        generated_flow_norm = ddpm_model.sample(
            batch_size=current_batch_size,
            hour_scalars_batch=hour_scalars, 
            is_holiday_scalars_batch=is_holiday_scalars 
        )
        
        generated_flow_denorm = generated_flow_norm * std_val + mean_val
        target_avg_flow_denorm = target_avg_flow_norm * std_val + mean_val
        all_generated_denorm_list.append(generated_flow_denorm.cpu())
        all_original_denorm_list.append(target_avg_flow_denorm.cpu())

        # samples_collected_fid = len(all_generated_samples_for_fid) * (dataloader.batch_size if dataloader.batch_size is not None else 0)
        samples_collected_fid = sum(s.shape[0] for s in all_generated_samples_for_fid)

        if samples_collected_fid < max_fid_samples_limit :
             remaining_needed = max_fid_samples_limit - samples_collected_fid
             samples_to_add = min(current_batch_size, remaining_needed)
             if samples_to_add > 0:
                 all_generated_samples_for_fid.append(generated_flow_norm[:samples_to_add].cpu())
                 all_original_samples_for_fid.append(target_avg_flow_norm[:samples_to_add].cpu())
    
    if not all_generated_denorm_list: 
        logger.warning("No data processed during evaluation. Returning zero/NaN metrics.")
        nan_grid = np.full((config["H"] * config["W"],), np.nan)
        return ({"mse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0, "fid": float('nan')},
                {'MSE': nan_grid, 'MAE': nan_grid, 'MAPE': nan_grid, 'SMAPE': nan_grid})

    generated_all_denorm_t = torch.cat(all_generated_denorm_list, dim=0)
    original_all_denorm_t = torch.cat(all_original_denorm_list, dim=0)
    epsilon = 1e-8 
    
    mse_total = F.mse_loss(generated_all_denorm_t, original_all_denorm_t).item()
    mae_total = F.l1_loss(generated_all_denorm_t, original_all_denorm_t).item()
    
    # MAPE and SMAPE calculations
    mape_tensor = torch.abs((original_all_denorm_t - generated_all_denorm_t) / (torch.abs(original_all_denorm_t) + epsilon)) * 100
    valid_mape_tensor = mape_tensor[torch.isfinite(mape_tensor)]
    mape_total = torch.mean(valid_mape_tensor).item() if valid_mape_tensor.numel() > 0 else float('inf')

    smape_numerator = torch.abs(generated_all_denorm_t - original_all_denorm_t)
    smape_denominator = torch.abs(original_all_denorm_t) + torch.abs(generated_all_denorm_t) + epsilon
    smape_tensor = 200 * smape_numerator / smape_denominator
    valid_smape_tensor = smape_tensor[torch.isfinite(smape_tensor)]
    smape_total = torch.mean(valid_smape_tensor).item() if valid_smape_tensor.numel() > 0 else float('inf')
    
    metrics: Dict[str, float] = {"mse": mse_total, "mae": mae_total, "mape": mape_total, "smape": smape_total, "fid": float('nan')}

    fid_score = float('nan')
    if all_generated_samples_for_fid and all_original_samples_for_fid:
        generated_tensor_fid = torch.cat(all_generated_samples_for_fid, dim=0)[:max_fid_samples_limit]
        original_tensor_fid = torch.cat(all_original_samples_for_fid, dim=0)[:max_fid_samples_limit]
        num_fid_samples_to_calc = min(generated_tensor_fid.shape[0], original_tensor_fid.shape[0])
        
        if num_fid_samples_to_calc > 1 : 
            logger.info(f"Calculating FID on {num_fid_samples_to_calc} samples...")
            try:
                act_generated = get_activations(generated_tensor_fid, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                act_original = get_activations(original_tensor_fid, inception_model_fid, config["device"], config.get("fid_batch_size", 64))
                if act_generated.shape[0] > 1 and act_original.shape[0] > 1:
                     fid_score = calculate_fid(act_original, act_generated)
                     logger.info(f"FID calculation completed: {fid_score:.4f}")
                else: logger.warning("Insufficient features for FID after activation.")
            except NameError as e: logger.error(f"FID: Function not defined? {e}")
            except Exception as e: logger.error(f"FID calculation failed: {e}")
        else: logger.warning(f"Insufficient samples ({num_fid_samples_to_calc}) for FID.")
    else: logger.warning("Sample lists for FID are empty.")
    metrics["fid"] = fid_score if np.isfinite(fid_score) else float('nan')

    logger.info("Generating detailed evaluation visualizations...")
    try:
        if generated_all_denorm_t.numel() > 0 and original_all_denorm_t.numel() > 0:
            visualize_predictions_long_term(generated_all_denorm_t.clone().cpu(), original_all_denorm_t.clone().cpu(), config, 0, "test_eval_sample0")
            visualize_predictions_long_term(generated_all_denorm_t.clone().cpu(), original_all_denorm_t.clone().cpu(), config, None, "test_eval_avg")
    except NameError: logger.error("visualize_predictions_long_term not defined.")
    except Exception as e: logger.error(f"Error in prediction visualization: {e}")

    error_metrics_grids: Dict[str, np.ndarray] = { m: np.full((config["H"] * config["W"],), np.nan) for m in ['MSE','MAE','MAPE','SMAPE']}
    if hasattr(dataset_obj, 'H') and hasattr(dataset_obj, 'W'): # type: ignore
        H_ds, W_ds = dataset_obj.H, dataset_obj.W # type: ignore
        num_grid_cells_ds = H_ds * W_ds
        if generated_all_denorm_t.ndim == 5 and generated_all_denorm_t.shape[-2:] == (H_ds, W_ds):
            mse_g = torch.mean((generated_all_denorm_t - original_all_denorm_t)**2, dim=(0,1,2)).cpu().numpy().flatten()
            mae_g = torch.mean(torch.abs(generated_all_denorm_t - original_all_denorm_t), dim=(0,1,2)).cpu().numpy().flatten()
            mape_g_tensor = torch.abs((original_all_denorm_t - generated_all_denorm_t) / (torch.abs(original_all_denorm_t) + epsilon)) * 100
            mape_g = torch.mean(mape_g_tensor, dim=(0,1,2)).cpu().numpy().flatten()
            smape_n_g = torch.abs(generated_all_denorm_t - original_all_denorm_t)
            smape_d_g = torch.abs(original_all_denorm_t) + torch.abs(generated_all_denorm_t) + epsilon
            smape_g_tensor = 200 * smape_n_g / smape_d_g
            smape_g = torch.mean(smape_g_tensor, dim=(0,1,2)).cpu().numpy().flatten()

            if len(mse_g) == num_grid_cells_ds:
                error_metrics_grids = {'MSE': mse_g, 'MAE': mae_g, 'MAPE': mape_g, 'SMAPE': smape_g}
                try: plot_grid_with_error_long_term(dataset_obj, error_metrics_grids, config, "test_eval") # type: ignore
                except NameError: logger.error("plot_grid_with_error_long_term not defined.")
                except Exception as e: logger.error(f"Error in grid error plotting: {e}")
            else: logger.warning(f"Per-grid metrics length mismatch. Skipping grid error plot.")
        else: logger.warning("Generated tensor shape mismatch. Skipping grid error plot.")
    else: logger.warning("Dataset missing H/W attributes. Skipping grid error plot.")
    logger.info("Detailed evaluation visualizations finished.")
    return metrics, error_metrics_grids
#%%
# 主訓練腳本
# (假設其定義與先前完整腳本相同，
# 但呼叫 ddpm.p_losses 的部分需傳遞純量小時/星期)
if __name__ == '__main__':
    logger.info("==========================================================")
    logger.info("    開始 DDPM 訓練 (額外資料未正規化，流量資料正規化)    ")
    logger.info("==========================================================")
    logger.info(f"組態設定: {json.dumps(CONFIG, indent=2)}")


    full_df = pd.read_csv(CONFIG["data_path"])
    logger.info(f"已載入資料: {CONFIG['data_path']}. 形狀: {full_df.shape}")



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


    logger.info("建立訓練資料集...")
    train_dataset = PeopleFlowDatasetCondition(train_df, CONFIG, mode='train')
    
    logger.info("建立驗證資料集...")
    val_dataset = PeopleFlowDatasetCondition(
        val_df,
        CONFIG,
        mode='val',
        average_flow_map_dict=train_dataset.average_flow_map_dict,
        norm_stats_flow=train_dataset.norm_stats_flow,
        sorted_flow_columns_from_train=train_dataset.sorted_flow_columns,
        grid_idx_to_rc_map_from_train=train_dataset.grid_idx_to_rc_map,
        processed_extra_columns_from_train=train_dataset.processed_extra_columns,
        selected_sensor_info_from_train=train_dataset.selected_sensor_info # <--- 確保傳遞此參數
    )

    logger.info("建立測試資料集...")
    test_dataset = PeopleFlowDatasetCondition(
        test_df,
        CONFIG,
        mode='test',
        average_flow_map_dict=train_dataset.average_flow_map_dict,
        norm_stats_flow=train_dataset.norm_stats_flow,
        sorted_flow_columns_from_train=train_dataset.sorted_flow_columns,
        grid_idx_to_rc_map_from_train=train_dataset.grid_idx_to_rc_map,
        processed_extra_columns_from_train=train_dataset.processed_extra_columns,
        selected_sensor_info_from_train=train_dataset.selected_sensor_info # <--- 確保傳遞此參數
    )


    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["eval_batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    logger.info("DataLoaders 建立完成。")

    logger.info("初始化 UNet3D 模型...")
    unet = UNet3D(
        CONFIG["image_channels"],
        CONFIG["base_channels_unet"],
        CONFIG["time_emb_dim"],
        CONFIG["condition_encode_dim"],
        dropout_rate=CONFIG.get("unet_dropout_rate", 0.05) 
    ).to(CONFIG["device"]) 
    logger.info("初始化 DDPM3D 模型...")
    ddpm = DDPM3D(unet, CONFIG["timesteps"], (CONFIG["D"],CONFIG["H"],CONFIG["W"]), CONFIG["image_channels"],
                  CONFIG["condition_input_channels"], CONFIG["condition_encode_dim"],
                  CONFIG["beta_start"], CONFIG["beta_end"], CONFIG["device"]).to(CONFIG["device"])
    logger.info(f"UNet3D 參數數量: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    logger.info(f"ConditionProcessor 參數數量: {sum(p.numel() for p in ddpm.condition_processor.parameters() if p.requires_grad):,}")

    # 優化器包含 U-Net 和條件處理器的參數
    optimizer = optim.AdamW(list(ddpm.model.parameters()) + list(ddpm.condition_processor.parameters()), lr=CONFIG["lr"])

    logger.info("載入 InceptionV3 以計算 FID...")
    # 保持 aux_logits=True 以匹配預訓練權重
    inception_fid = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)

    inception_fid.fc = nn.Identity()

    # 按照您原始程式碼的風格，如果存在 AuxLogits，則將其設為 None
    if hasattr(inception_fid, 'AuxLogits') and inception_fid.AuxLogits is not None:
        inception_fid.AuxLogits = None

    inception_fid = inception_fid.to(CONFIG["device"])
    inception_fid.eval()
    logger.info("InceptionV3 載入完成。")

    optimizer = optim.AdamW(
    list(ddpm.model.parameters()) + list(ddpm.condition_processor.parameters()),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]
)

# --- 在開始訓練迴圈之前，定義 scheduler ---
# (修改) Scheduler 現在監控 avg_train_loss
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min', # 訓練損失越小越好
                              factor=CONFIG["lr_scheduler_factor"],
                              patience=CONFIG["lr_scheduler_patience"],
                              min_lr=CONFIG["lr_scheduler_min_lr"])

start_epoch = 1 # 預設從 epoch 1 開始
best_val_loss_for_saving = float('inf')
best_val_loss_epoch = 0
metrics_hist = {'train_loss':[], 'val_loss':[], 'lr':[]}
early_stopping_counter = 0
last_calculated_avg_val_loss = float('inf')

checkpoint_filename = CONFIG.get("checkpoint_path", "best_ddpm_model_during_training.pth")
checkpoint_full_path = os.path.join(CONFIG["save_dir"], checkpoint_filename)

if CONFIG.get("resume_from_checkpoint", True) and os.path.exists(checkpoint_full_path):
    logger.info(f"找到檢查點: {checkpoint_full_path}，嘗試載入...")
    try:
        # 使用之前解決 UnpicklingError 的方法載入
        import numpy
        import pickle
        with torch.serialization.safe_globals([numpy, numpy.float32, numpy.float64, numpy.int32, numpy.int64]):
            chkpt = torch.load(checkpoint_full_path, map_location=CONFIG["device"], weights_only=False)
        
        # 載入模型狀態
        ddpm.load_state_dict(chkpt['ddpm_state_dict'])
        
        # 載入優化器和排程器狀態 (如果存在)
        if 'optimizer_state_dict' in chkpt:
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            logger.info("已成功載入優化器狀態。")
        else:
            logger.warning("檢查點中未找到 'optimizer_state_dict'，優化器將從頭開始。")

        if 'scheduler_state_dict' in chkpt:
            scheduler.load_state_dict(chkpt['scheduler_state_dict'])
            logger.info("已成功載入排程器狀態。")
        else:
            logger.warning("檢查點中未找到 'scheduler_state_dict'，排程器將從頭開始。")

        # 恢復訓練進度相關的變數
        start_epoch = chkpt.get('epoch', 0) + 1 # 從下一個 epoch 開始
        
        # 恢復最佳驗證損失 (用於模型保存)
        best_val_loss_for_saving = chkpt.get('best_val_loss_for_saving', float('inf'))
        best_val_loss_epoch = chkpt.get('epoch', 0) # epoch ที่บันทึก best_val_loss
        
        # 比較儲存的CONFIG和當前的CONFIG (可選，但建議)
        saved_config = chkpt.get('config', None)
        if saved_config:
            # 這裡可以加入更詳細的 CONFIG 比較邏輯
            if saved_config['H'] != CONFIG['H'] or saved_config['W'] != CONFIG['W']:
                logger.warning("警告：載入的檢查點 CONFIG 與當前 CONFIG 的網格尺寸不符！可能導致錯誤。")
            # ... 可以比較更多關鍵參數 ...
        
        logger.info(f"成功從 epoch {start_epoch-1} 的檢查點恢復訓練。將從 epoch {start_epoch} 開始。")
        logger.info(f"恢復的最佳驗證損失 (用於模型保存): {best_val_loss_for_saving:.5f} (在 epoch {best_val_loss_epoch})")

    except Exception as e:
        logger.error(f"載入檢查點 {checkpoint_full_path} 失敗: {e}。將從頭開始訓練。")
        start_epoch = 1 # 確保如果載入失敗，從頭開始
        # 重置其他可能被部分修改的變數
        best_val_loss_for_saving = float('inf')
        best_val_loss_epoch = 0
        metrics_hist = {'train_loss':[], 'val_loss':[], 'lr':[]}
        early_stopping_counter = 0
        last_calculated_avg_val_loss = float('inf')
else:
    logger.info("未找到檢查點或未設定從檢查點恢復。將從頭開始訓練。")
    # start_epoch 等變數已是預設值
#%%
logger.info("開始訓練迴圈...")

# 最佳驗證損失，用於 scheduler 和模型保存
best_val_loss = float('inf')
best_val_loss_epoch = 0

metrics_hist = {'train_loss':[], 'val_loss':[], 'lr':[]}

early_stopping_patience = CONFIG["early_stopping_patience"]
early_stopping_counter = 0


for epoch in range(1, CONFIG["epochs"] + 1):
    # --- 訓練階段 ---
    ddpm.train()
    total_train_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [訓練]", leave=False)
    for x_start, hour_s, is_holiday_s, _ in train_pbar:
        optimizer.zero_grad()
        x_start = x_start.to(CONFIG["device"])
        t = torch.randint(0, CONFIG["timesteps"], (x_start.shape[0],), device=CONFIG["device"]).long()
        loss = ddpm.p_losses(x_start, t, hour_s, is_holiday_s)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_pbar.set_postfix({"損失": loss.item()})

    avg_train_loss = total_train_loss / len(train_loader)
    metrics_hist['train_loss'].append(avg_train_loss)

    # --- 驗證階段 (每個 Epoch 都執行) ---
    ddpm.eval()
    total_val_loss = 0
    
    if len(val_loader.dataset) > 0:
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [驗證]", leave=False)
            for val_x_start, val_hour_s, val_is_holiday_s, _ in val_pbar:
                val_x_start = val_x_start.to(CONFIG["device"])
                t = torch.randint(0, CONFIG["timesteps"], (val_x_start.shape[0],), device=CONFIG["device"]).long()
                batch_val_loss = ddpm.p_losses(val_x_start, t, val_hour_s, val_is_holiday_s)
                total_val_loss += batch_val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
    else:
        avg_val_loss = float('inf') # 若驗證集為空，設為無效值

    metrics_hist['val_loss'].append(avg_val_loss)
    
    # --- 更新、日誌、儲存與早停 ---
    
    # 使用驗證損失來更新學習率排程器
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    metrics_hist['lr'].append(current_lr)
    
    val_loss_display = f"{avg_val_loss:.5f}" if avg_val_loss != float('inf') else "N/A"
    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss_display} | LR: {current_lr:.8f}")

    # 使用驗證損失來判斷是否儲存最佳模型與早停
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_loss_epoch = epoch
        early_stopping_counter = 0
        
        save_path = os.path.join(CONFIG["save_dir"], "best_ddpm_model_during_training.pth")
        torch.save({
            'epoch': epoch,
            'ddpm_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss_at_best_val': avg_train_loss,
            'config': CONFIG,
            'norm_stats_flow': train_dataset.norm_stats_flow,
            'sorted_flow_columns': train_dataset.sorted_flow_columns,
            'grid_idx_to_rc_map': train_dataset.grid_idx_to_rc_map,
            'selected_sensor_info': train_dataset.selected_sensor_info,
            'processed_extra_columns': train_dataset.processed_extra_columns,
        }, save_path)
        logger.info(f"已儲存新的最佳模型 (Epoch {best_val_loss_epoch} based on Val Loss: {best_val_loss:.5f})")
    else:
        early_stopping_counter += 1
        logger.info(f"驗證損失未改善，早停計數: {early_stopping_counter}/{early_stopping_patience}")
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"早停機制觸發於 Epoch {epoch}。")
            break

logger.info("訓練完成。")
#%%
    # --- 所有 epoch 訓練完成後，載入最佳模型並進行最終評估 ---
logger.info("載入訓練過程中驗證損失最低的模型以進行最終測試集評估...")
best_model_path = os.path.join(CONFIG["save_dir"], "best_ddpm_model_during_training.pth")

if not os.path.exists(best_model_path):
    logger.error(f"找不到在訓練過程中儲存的最佳模型檔案: {best_model_path}。將使用最後一個 epoch 的模型進行評估。")
    # 如果沒有找到最佳模型（例如，如果上面的儲存邏輯有問題或被跳過），
    # final_ddpm 會是訓練結束時的 ddpm 狀態。
    # 也可以選擇在這裡 raise Error 或採取其他策略。
    final_ddpm_for_eval = ddpm # 使用最後一個 epoch 的模型
else:
    chkpt = torch.load(best_model_path, map_location=CONFIG["device"], weights_only=False)
    cfg_chkpt = chkpt.get('config', CONFIG)

    # 重新初始化模型結構以載入狀態字典
    final_unet_for_eval = UNet3D(
        cfg_chkpt["image_channels"],
        cfg_chkpt["base_channels_unet"],
        cfg_chkpt["time_emb_dim"],
        cfg_chkpt["condition_encode_dim"]
    ).to(CONFIG["device"])

    final_ddpm_for_eval = DDPM3D(
        final_unet_for_eval,
        cfg_chkpt["timesteps"],
        (cfg_chkpt["D"], cfg_chkpt["H"], cfg_chkpt["W"]),
        cfg_chkpt["image_channels"],
        cfg_chkpt["condition_input_channels"],
        cfg_chkpt["condition_encode_dim"],
        beta_start=cfg_chkpt.get("beta_start", CONFIG["beta_start"]),
        beta_end=cfg_chkpt.get("beta_end", CONFIG["beta_end"]),
        device=CONFIG["device"]
    )
    final_ddpm_for_eval.load_state_dict(chkpt['ddpm_state_dict'])
    logger.info(f"從 {best_model_path} 載入最佳模型 (Epoch {chkpt.get('epoch', '未知')}) 完成。")

# 使用 test_loader 和載入的最佳模型 (final_ddpm_for_eval) 進行最終評估
logger.info("在測試集上評估載入的最佳模型...")
# 確保 inception_fid 模型已定義和載入
if 'inception_fid' not in locals() or inception_fid is None:
    logger.info("重新載入 InceptionV3 以計算 FID (因為可能在訓練迴圈中未持續保持)...")
    inception_fid = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    inception_fid.fc = nn.Identity()
    if hasattr(inception_fid, 'AuxLogits') and inception_fid.AuxLogits is not None:
        inception_fid.AuxLogits = None
    inception_fid = inception_fid.to(CONFIG["device"])
    inception_fid.eval()
    logger.info("InceptionV3 載入完成。")

test_metrics, per_grid_test_metrics = evaluate_model(
        final_ddpm_for_eval, 
        test_loader, 
        inception_fid, 
        CONFIG, 
        CONFIG["fid_num_samples"]
    )
logger.info(f"最終測試結果: MSE:{test_metrics['mse']:.5f}|MAE:{test_metrics['mae']:.5f}|MAPE:{test_metrics['mape']:.2f}%|SMAPE:{test_metrics['smape']:.2f}%|FID:{test_metrics['fid']:.3f}")

# 儲存最終測試指標
with open(os.path.join(CONFIG["save_dir"], "final_test_metrics.json"),'w') as f:
    json.dump(test_metrics, f, indent=4)
with open(os.path.join(CONFIG["save_dir"], "final_test_metrics.txt"),'w') as f:
    f.write(f"FINAL TEST METRICS:\nDate: {pd.Timestamp.now(tz='Asia/Taipei')}\n")
    for k,v in test_metrics.items():
        f.write(f"{k.upper()}: {v:.6f}\n")

logger.info("開始準備匯出 Excel 檔案的詳細指標...")
H_test = CONFIG["H"]
W_test = CONFIG["W"]
num_grid_cells_test = H_test * W_test

excel_data_rows = []

# test_dataset 可以從 test_loader 獲得
current_test_dataset = test_loader.dataset 

# 準備感測器資訊以便快速查找經緯度
sensor_info_lookup = {info['name']: {'lon': info['lon'], 'lat': info['lat']}
                        for info in current_test_dataset.selected_sensor_info}

for flat_idx in range(num_grid_cells_test):
    grid_rc = current_test_dataset.grid_idx_to_rc_map.get(flat_idx, (-1, -1)) # (row, col)
    
    lon, lat = np.nan, np.nan # 預設為 NaN
    if flat_idx < len(current_test_dataset.sorted_flow_columns):
        col_name = current_test_dataset.sorted_flow_columns[flat_idx]
        if col_name in sensor_info_lookup:
            lon = sensor_info_lookup[col_name]['lon']
            lat = sensor_info_lookup[col_name]['lat']
        else:
            logger.warning(f"Excel匯出：在 sensor_info_lookup 中找不到欄位 {col_name} (網格索引 {flat_idx}) 的經緯度。")
    else:
        logger.warning(f"Excel匯出：網格索引 {flat_idx} 超出 sorted_flow_columns 的範圍。")

    row_data = {
        '網格座標_R': grid_rc[0] if grid_rc[0] != -1 else '', # 網格橫座標
        '網格座標_C': grid_rc[1] if grid_rc[1] != -1 else '', # 網格縱座標
        '經度': lon,
        '緯度': lat,
        'MSE': per_grid_test_metrics.get('MSE')[flat_idx] if per_grid_test_metrics.get('MSE') is not None and flat_idx < len(per_grid_test_metrics.get('MSE')) else np.nan,
        'MAE': per_grid_test_metrics.get('MAE')[flat_idx] if per_grid_test_metrics.get('MAE') is not None and flat_idx < len(per_grid_test_metrics.get('MAE')) else np.nan,
        'MAPE': per_grid_test_metrics.get('MAPE')[flat_idx] if per_grid_test_metrics.get('MAPE') is not None and flat_idx < len(per_grid_test_metrics.get('MAPE')) else np.nan,
        'SMAPE': per_grid_test_metrics.get('SMAPE')[flat_idx] if per_grid_test_metrics.get('SMAPE') is not None and flat_idx < len(per_grid_test_metrics.get('SMAPE')) else np.nan,
        'FID': 'N/A' # FID 通常不是針對每個網格單元計算的
    }
    excel_data_rows.append(row_data)

# 準備平均指標列 (最後一列)
average_row_data = {
    '網格座標_R': '整體平均',
    '網格座標_C': '',
    '經度': '',
    '緯度': '',
    'MSE': test_metrics.get('mse', np.nan),
    'MAE': test_metrics.get('mae', np.nan),
    'MAPE': test_metrics.get('mape', np.nan),
    'SMAPE': test_metrics.get('smape', np.nan),
    'FID': test_metrics.get('fid', np.nan) # 全域 FID
}
excel_data_rows.append(average_row_data)

df_excel = pd.DataFrame(excel_data_rows)

# 定義 Excel 中的欄位順序
excel_column_order = ['網格座標_R', '網格座標_C', '經度', '緯度', 'MSE', 'MAE', 'MAPE', 'SMAPE', 'FID']
df_excel = df_excel[excel_column_order]

excel_filename = "final_test_metrics_detailed.xlsx"
excel_save_path = os.path.join(CONFIG["save_dir"], excel_filename)

try:
    df_excel.to_excel(excel_save_path, index=False, sheet_name='詳細測試指標')
    logger.info(f"詳細測試指標已成功匯出至 Excel 檔案: {excel_save_path}")
except Exception as e:
    logger.error(f"匯出 Excel 檔案失敗: {e}")

# 繪製訓練歷史圖表 (只包含訓練損失和驗證損失)
num_train_epochs_recorded = len(metrics_hist.get('train_loss', []))
num_val_epochs_recorded = len(metrics_hist.get('val_loss', []))
num_epochs_to_plot = min(num_train_epochs_recorded, num_val_epochs_recorded)

if num_epochs_to_plot > 0:
    ep_rng_plot = range(1, num_epochs_to_plot + 1)
    train_loss_plot = metrics_hist['train_loss'][:num_epochs_to_plot]
    val_loss_plot = metrics_hist['val_loss'][:num_epochs_to_plot]

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.plot(ep_rng_plot, train_loss_plot, label='Training Loss')
    plt.plot(ep_rng_plot, val_loss_plot, label='Validation Loss (MSE)')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'Training and Validation Loss History (up to {num_epochs_to_plot} epochs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_loss_history_plot.png"))
    plt.close()
    logger.info(f"已儲存訓練和驗證損失歷史圖表 (繪製了 {num_epochs_to_plot} 個 epochs)。")
else:
    logger.info("沒有足夠的數據來繪製訓練和驗證損失歷史圖表 (可能由於訓練提早中斷)。")

logger.info("================ 腳本執行完成 ================")
# %%

# -*- coding: utf-8 -*-
"""
功能:
1. 讀取人流資料CSV檔案。
2. 在指定時間範圍內，找出總人數最多的時刻。
3. 在指定時間範圍且符合額外條件（如雨天）的數據中，找出總人數最少的時刻。
4. 將這兩個特定小時的人群地理分布繪製成圖並儲存。
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 腳本組態設定
# ==============================================================================
CONFIG = {
    # --- 路徑設定 ---
    "data_path": r"C:\thesis\code\Taipei_CF\all_merged.csv",
    "output_dir": "min_max_hour_plots_final",

    # --- 全域時間過濾設定 ---
    "time_filter": {
        "enabled": True,
        "start_hour_inclusive": 8,
        "end_hour_inclusive": 22
    },

    # --- 【新增】尋找最少人數時的額外條件 ---
    "min_flow_conditions": {
        "enabled": True,
        "column_name": "降水量", # 根據您的 CSV 欄位名稱
        "operator": ">",
        "value": 0
    },

    # --- 繪圖設定 ---
    "plot_filenames": {
        "max_hour": "max_flow_hour_daytime.png",
        "min_hour": "min_flow_hour_rainy_daytime.png" # 更新檔名以反映條件
    },
    "plot_config": {
        "cmap": "viridis",
        "marker_size": 100
    }
}

# ==============================================================================
# 輔助函式 (與前版相同)
# ==============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_lat_lon(column_name: str) -> Tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(2)), float(match.group(1))
    return None

def plot_flow_map(flow_data: np.ndarray, coords: List[Tuple[float, float]], title: str, output_path: str, config: Dict[str, Any], vmin: float, vmax: float):
    logger = logging.getLogger(__name__)
    lats, lons = zip(*coords)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(lons, lats, c=flow_data, cmap=config['cmap'], s=config['marker_size'], marker='s', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('People Count', rotation=270, labelpad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Map saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save map: {e}")
    finally:
        plt.close(fig)

# ==============================================================================
# 主執行流程
# ==============================================================================
def main():
    """主執行函式"""
    logger = setup_logging()

    logger.info("===== Starting Min/Max Flow Hour Analysis Script =====")
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 讀取並準備資料 ---
    data_path = CONFIG["data_path"]
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"ERROR: Data file not found at '{data_path}'.")
        return
        
    df['時間'] = pd.to_datetime(df['時間'])
    df['hour'] = df['時間'].dt.hour
    
    flow_columns = [col for col in df.columns if parse_lat_lon(col) is not None]
    if not flow_columns:
        logger.error("ERROR: No valid coordinate-formatted columns found.")
        return
    logger.info(f"Successfully identified {len(flow_columns)} people flow columns.")
    
    coordinates = [parse_lat_lon(col) for col in flow_columns]

    # --- 2. 應用全域時間過濾 ---
    time_filter_config = CONFIG["time_filter"]
    if time_filter_config["enabled"]:
        start_h, end_h = time_filter_config["start_hour_inclusive"], time_filter_config["end_hour_inclusive"]
        logger.info(f"Applying time filter: only including hours from {start_h}:00 to {end_h}:00.")
        df_daytime = df[(df['hour'] >= start_h) & (df['hour'] <= end_h)].copy()
    else:
        df_daytime = df.copy()

    if df_daytime.empty:
        logger.error("No data remains after applying the time filter. Exiting.")
        return

    # --- 3. 計算總人數 ---
    logger.info("Calculating total flow for each valid hour...")
    df_daytime['total_flow'] = df_daytime[flow_columns].sum(axis=1)
    
    # --- 4. 尋找最多人流的時刻 (在所有日間數據中) ---
    max_idx = df_daytime['total_flow'].idxmax()
    max_hour_series = df_daytime.loc[max_idx]
    max_timestamp = max_hour_series.get('時間', 'N/A')
    logger.info(f"Max flow hour found: {max_timestamp} (Total Flow: {max_hour_series['total_flow']:.0f})")
    max_flow_data = max_hour_series[flow_columns].values

    # --- 5. 【核心修改】尋找最少人流的時刻 (在日間且為雨天的數據中) ---
    min_flow_cond_config = CONFIG["min_flow_conditions"]
    df_for_min_search = df_daytime.copy()

    if min_flow_cond_config["enabled"]:
        col = min_flow_cond_config["column_name"]
        op = min_flow_cond_config["operator"]
        val = min_flow_cond_config["value"]
        
        if col not in df_for_min_search.columns:
            logger.error(f"Error: Column '{col}' for min-flow condition not found in data. Skipping this condition.")
        else:
            logger.info(f"Applying extra condition for min-flow search: '{col} {op} {val}'")
            if op == '>':
                df_for_min_search = df_for_min_search[df_for_min_search[col] > val]
            # 可在此處加入更多運算符的判斷 (如 <, ==)
            else:
                logger.error(f"Unsupported operator '{op}' in min_flow_conditions.")
    
    if df_for_min_search.empty:
        logger.warning("No data meets the criteria for finding the minimum flow hour (e.g., no rainy days in the daytime dataset). Skipping min-flow plot.")
        min_flow_data = None
    else:
        min_idx = df_for_min_search['total_flow'].idxmin()
        min_hour_series = df_for_min_search.loc[min_idx]
        min_timestamp = min_hour_series.get('時間', 'N/A')
        logger.info(f"Min flow hour (under conditions) found: {min_timestamp} (Total Flow: {min_hour_series['total_flow']:.0f})")
        min_flow_data = min_hour_series[flow_columns].values

    # --- 6. 準備繪圖 ---
    # 決定統一的顏色標尺
    vmin = np.min(min_flow_data) if min_flow_data is not None else np.min(max_flow_data)
    vmax = np.max(max_flow_data)
    logger.info(f"Using consistent color scale for plots: min={vmin:.0f}, max={vmax:.0f}")

    # 繪製人數最多小時的地圖
    plot_flow_map(
        flow_data=max_flow_data, coords=coordinates,
        title=f"Maximum People Flow Hour (Daytime)\n{max_timestamp}",
        output_path=os.path.join(output_dir, CONFIG["plot_filenames"]["max_hour"]),
        config=CONFIG["plot_config"], vmin=vmin, vmax=vmax
    )

    # 如果找到了符合條件的最少人數時刻，才進行繪圖
    if min_flow_data is not None:
        plot_flow_map(
            flow_data=min_flow_data, coords=coordinates,
            title=f"Minimum People Flow Hour (Rainy Daytime)\n{min_timestamp}",
            output_path=os.path.join(output_dir, CONFIG["plot_filenames"]["min_hour"]),
            config=CONFIG["plot_config"], vmin=vmin, vmax=vmax
        )

    logger.info("===== All tasks completed successfully =====")

if __name__ == '__main__':
    main()
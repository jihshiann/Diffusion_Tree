# -*- coding: utf-8 -*-
"""
功能:
1. 計算每個網格的全時段平均人流作為基準。
2. 根據平日與週末，分別計算其平均人流。
3. 將兩個條件的平均人流，分別除以各自網格的基準平均，得到一個相對強度比例。
4. 將兩個比例結果繪製成兩張獨立的地圖，每張圖使用其自身的顏色標尺。
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ==============================================================================
# 腳本組態設定
# ==============================================================================
CONFIG = {
    # --- 路徑設定 ---
    "data_path": r"C:\thesis\code\Taipei_CF\all_merged.csv",
    "output_dir": "relative_flow_weekday_weekend_analysis",

    # --- 繪圖檔名設定 ---
    "plot_filenames": {
        "weekday": "relative_flow_weekday.png",
        "weekend": "relative_flow_weekend.png"
    },
    
    # --- 繪圖參數 ---
    "plot_config": {
        "cmap": "inferno",  # 【已更新】使用 'inferno' 色彩主題
        "marker_size": 100
    },
    
    # --- 計算參數 ---
    "epsilon": 1e-6
}

# ==============================================================================
# 輔助函式
# ==============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_lat_lon(column_name: str) -> Tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(2)), float(match.group(1))
    return None

def plot_relative_flow_map(ratio_data: np.ndarray, coords: List[Tuple[float, float]], title: str, output_path: str):
    """繪製相對強度比例圖，顏色標尺由數據本身決定"""
    logger = logging.getLogger(__name__)
    lats, lons = zip(*coords)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    
    vmin = np.min(ratio_data)
    vmax = np.max(ratio_data)
    logger.info(f"為圖表 '{title}' 設定獨立顏色標尺: min={vmin:.2f}, max={vmax:.2f}")

    scatter = ax.scatter(lons, lats, c=ratio_data, cmap=CONFIG["plot_config"]["cmap"], s=CONFIG["plot_config"]["marker_size"], marker='s', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative Flow Intensity (vs. Grid Average)')
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"相對強度地圖已儲存至: {output_path}")
    finally:
        plt.close(fig)

# ==============================================================================
# 主執行流程
# ==============================================================================
def main():
    logger = setup_logging()
    logger.info("===== 開始執行平日與週末相對人流強度分析腳本 =====")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # --- 1. 讀取並準備資料 ---
    try:
        df = pd.read_csv(CONFIG["data_path"])
    except FileNotFoundError:
        logger.error(f"錯誤：找不到資料檔案 '{CONFIG['data_path']}'。")
        return
        
    df['時間'] = pd.to_datetime(df['時間'])
    df['weekday'] = df['時間'].dt.dayofweek # 星期一=0, ..., 星期日=6
    
    flow_columns = [col for col in df.columns if parse_lat_lon(col) is not None]
    if not flow_columns:
        logger.error("錯誤：在資料中找不到任何有效的座標格式欄位。")
        return
    coordinates = [parse_lat_lon(col) for col in flow_columns]
    logger.info(f"成功解析出 {len(flow_columns)} 個流量網格點。")

    # --- 2. 計算每個網格的全時段平均人流 (基準) ---
    grid_overall_avg_flow = df[flow_columns].values.mean(axis=0)
    logger.info(f"計算完成每個網格的全時段平均人流 (基準)。")

    # --- 3. 計算 平日 (weekday <= 4) 的平均人流與相對強度 ---
    df_weekday = df[df['weekday'] <= 4]
    if df_weekday.empty:
        logger.error("找不到任何平日 (weekday <= 4) 的數據，無法繼續。")
        return
    weekday_avg_flow = df_weekday[flow_columns].values.mean(axis=0)
    ratio_weekday = weekday_avg_flow / (grid_overall_avg_flow + CONFIG['epsilon'])
    logger.info(f"計算完成平日相對人流強度。")

    # --- 4. 計算 週末 (weekday > 4) 的平均人流與相對強度 ---
    df_weekend = df[df['weekday'] > 4]
    if df_weekend.empty:
        logger.error("找不到任何週末 (weekday > 4) 的數據，無法繼續。")
        return
    weekend_avg_flow = df_weekend[flow_columns].values.mean(axis=0)
    ratio_weekend = weekend_avg_flow / (grid_overall_avg_flow + CONFIG['epsilon'])
    logger.info(f"計算完成週末相對人流強度。")
    
    # --- 5. 繪製兩張圖 ---
    
    # 繪製平日的相對強度圖
    plot_relative_flow_map(
        ratio_data=ratio_weekday,
        coords=coordinates,
        title="Relative Flow Intensity (Weekdays vs. Grid Average)",
        output_path=os.path.join(CONFIG["output_dir"], CONFIG["plot_filenames"]["weekday"])
    )

    # 繪製週末的相對強度圖
    plot_relative_flow_map(
        ratio_data=ratio_weekend,
        coords=coordinates,
        title="Relative Flow Intensity (Weekend vs. Grid Average)",
        output_path=os.path.join(CONFIG["output_dir"], CONFIG["plot_filenames"]["weekend"])
    )

    logger.info("===== 所有任務執行完畢 =====")

if __name__ == '__main__':
    main()
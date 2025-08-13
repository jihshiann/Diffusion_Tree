# -*- coding: utf-8 -*-
"""
功能:
1. 讀取人流資料。
2. 產生一張包含兩個子圖的 KDE 比較圖。
3. 在 Min-Max 圖上同時標示出「中間點(0.5)」和「真實平均值的位置」。
4. 在 Z-score 圖上標示出「平均值(0)」。
5. 清晰地視覺化 Z-score 的0點作為數據重心的統計意義。
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

# ==============================================================================
# 腳本組態設定
# ==============================================================================
CONFIG = {
    "data_path": r"C:\thesis\code\Taipei_CF\all_merged.csv",
    "output_dir": "distribution_plots_output",
    "output_filename": "scaling_meaning_comparison.png",
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

# ==============================================================================
# 主執行流程
# ==============================================================================
def main():
    logger = setup_logging()
    logger.info("===== Starting Scaling Method's Statistical Meaning Comparison Script =====")
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    data_path = CONFIG["data_path"]
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"ERROR: Data file not found at '{data_path}'.")
        return

    flow_columns = [col for col in df.columns if parse_lat_lon(col) is not None]
    if not flow_columns:
        logger.error("ERROR: No valid coordinate-formatted columns found.")
        return
        
    all_flow_data = df[flow_columns].values.flatten()
    all_flow_data = all_flow_data[~np.isnan(all_flow_data)]
    
    # --- 1. 計算整批資料的統計數據 ---
    mean_val = np.mean(all_flow_data)
    std_val = np.std(all_flow_data)
    min_val = np.min(all_flow_data)
    max_val = np.max(all_flow_data)
    
    logger.info(f"Full dataset stats: Mean={mean_val:.2f}, Std={std_val:.2f}, Min={min_val:.2f}, Max={max_val:.2f}")

    # --- 2. 對數據進行兩種標準化 ---
    zscore_data = (all_flow_data - mean_val) / std_val if std_val > 1e-6 else np.zeros_like(all_flow_data)
    minmax_data = (all_flow_data - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else np.zeros_like(all_flow_data)
    
    # --- 3. 計算原始平均值在 Min-Max 尺度上的對應位置 ---
    minmax_mean_pos = (mean_val - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else 0
    
    # --- 4. 繪製比較圖 ---
    logger.info("Plotting KDE distributions with mean/midpoint indicators...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Visualizing the Statistical Meaning of Scaling', fontsize=18)

    # --- 子圖 1: Min-Max Scaling ---
    sns.kdeplot(minmax_data, ax=axes[0], fill=True, clip=(0, 1))
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label=f'Midpoint = 0.5')
    axes[0].axvline(minmax_mean_pos, color='green', linestyle=':', linewidth=2, label=f'Position of Mean = {minmax_mean_pos:.2f}')
    axes[0].set_title('Min-Max Scaled Distribution')
    axes[0].set_xlabel('Scaled Value [0, 1]')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # --- 子圖 2: Z-score Normalization ---
    sns.kdeplot(zscore_data, ax=axes[1], fill=True, color='darkorange')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
    axes[1].set_title('Z-score Normalized Distribution')
    axes[1].set_xlabel('Z-score Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    # 限制X軸範圍以更好地觀察中心區域
    axes[1].set_xlim(np.percentile(zscore_data, 0.1), np.percentile(zscore_data, 99))


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # --- 5. 儲存圖片 ---
    output_path = os.path.join(output_dir, CONFIG["output_filename"])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Meaning comparison chart saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save meaning comparison chart: {e}")
    finally:
        plt.close()

    logger.info("===== All tasks completed successfully =====")

if __name__ == '__main__':
    main()
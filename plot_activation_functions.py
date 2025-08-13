# -*- coding: utf-8 -*-
"""
功能:
1. 定義 ReLU 和 SiLU (Swish) 激活函數。
2. 在同一個圖表上繪製這兩個函數的曲線。
3. 標示出兩者在平滑性、對負值的處理等方面的關鍵差異。
4. 將比較圖儲存為 PNG 檔案。
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# ==============================================================================
# 腳本組態設定
# ==============================================================================
CONFIG = {
    "output_dir": "activation_plots_output",
    "output_filename": "relu_vs_silu_comparison.png",
    "x_range": [-5, 5], # 繪圖的X軸範圍
    "num_points": 400   # 繪圖的點密度，越高越平滑
}

# ==============================================================================
# 輔助函式
# ==============================================================================
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

# 定義激活函數
def relu(x):
    """ReLU 函數"""
    return np.maximum(0, x)

def silu(x):
    """SiLU / Swish 函數"""
    return x / (1 + np.exp(-x))

# ==============================================================================
# 主執行流程
# ==============================================================================
def main():
    logger = setup_logging()

    # --- 1. 準備環境和數據 ---
    logger.info("===== Starting Activation Function Comparison Plot Script =====")
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 產生 X 軸的數值
    x = np.linspace(CONFIG["x_range"][0], CONFIG["x_range"][1], CONFIG["num_points"])
    
    # 計算對應的 Y 軸數值
    y_relu = relu(x)
    y_silu = silu(x)
    
    # --- 2. 繪製圖表 ---
    logger.info("Plotting ReLU and SiLU functions...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    
    # 繪製兩條曲線
    plt.plot(x, y_relu, label='ReLU', color='dodgerblue', linewidth=2.5)
    plt.plot(x, y_silu, label='SiLU / Swish', color='darkorange', linewidth=2.5)
    
    # 繪製輔助線
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.axvline(0, color='gray', linewidth=1, linestyle='--')
    
    # --- 3. 加上註解以凸顯差異 ---
    # 標示 ReLU 的尖角
    plt.annotate('Sharp corner at x=0', xy=(0, 0), xytext=(0.5, 1.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=12, color='dodgerblue')

    # 標示 SiLU 的平滑曲線與負值區間
    plt.annotate('Smooth curve, dips below zero', xy=(-1.2, -0.2), xytext=(-4, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=12, color='darkorange')

    # --- 4. 美化與儲存圖表 ---
    plt.title('ReLU vs. SiLU (Swish) Activation Functions', fontsize=16)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (f(x))')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle=':', linewidth=0.8)
    plt.axis([CONFIG["x_range"][0], CONFIG["x_range"][1], -1, 4]) # 設定座標軸顯示範圍

    output_path = os.path.join(output_dir, CONFIG["output_filename"])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison chart saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save chart: {e}")
    finally:
        plt.close()

    logger.info("===== All tasks completed successfully =====")

if __name__ == '__main__':
    main()
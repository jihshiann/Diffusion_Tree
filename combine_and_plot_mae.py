# -*- coding: utf-8 -*-
"""
功能:
1. 從多個 Stage4 Excel 報告中讀取並整合 MAE 數據。
2. 根據 MAE 最低原則合併數據，並用 Stage3 數據填補空缺。
3. 將網格數據轉換為地理座標。
4. 繪製一張地理分布圖，其中 Stage4 數據有彩色外框，Stage3 填補的數據無外框。
5. 儲存最終的視覺化結果。
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Any

# ==============================================================================
# 腳本組態設定 (請在此處修改所有參數)
# ==============================================================================
CONFIG = {
    # --- 1. 輸入的 Stage4 Excel 報告路徑 (可新增多個) ---
    "input_excel_paths": [
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\stage4_ArenaEvents\final_eval_filtered_comparison_report.xlsx",
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\stage4_ArenaEvents_concert\final_eval_filtered_comparison_report.xlsx",
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\Stage4_RelativeHumidityLe675\final_eval_filtered_comparison_report.xlsx",
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\Stage4_MonthMe10\final_eval_filtered_comparison_report.xlsx",
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\Stage4_MonthLe5\final_eval_filtered_comparison_report.xlsx",
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\Stage4_UVIndexM0\final_eval_filtered_comparison_report.xlsx",
        r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage4\Stage4_MonthLe2\final_eval_filtered_comparison_report.xlsx",
    ],

    # --- 2. 用於填補空缺的 Stage3 Excel 報告路徑 ---
    "filler_excel_path": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_stage3\Stage3_WeekdayLe4\final_s3_evaluation_metrics_detailed.xlsx",

    # --- 3. 匈牙利演算法網格映射表路徑 ---
    "mapping_table_path": r"C:\thesis\code\DIFFUSION_TREE\results_ddpm_long-term\hungarian_grid_mapping_table.xlsx",

    # --- 4. 輸出設定 ---
    "output_dir": "mae_combination_output",
    "output_filename": "combined_mae_map_no_s3_outline.png",

    # --- 5. 網格與繪圖參數 ---
    "grid_shape": (20, 20),
    "plot_config": {
        "marker_size": 100,
        "outline_linewidth": 2.0,
        "fill_color": "white",
        "text_label": {
            "enabled": True,
            "fontsize": 6,
            "color": "black"
        }
    }
}

# ==============================================================================
# 輔助函式 (與前版相同)
# ==============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def get_source_label(filepath: str) -> str:
    return os.path.basename(os.path.dirname(filepath))

def load_data_from_excel(filepath: str, source_filter: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_excel(filepath, sheet_name=0)
        logger.info(f"成功讀取檔案: {os.path.basename(filepath)}")
        
        df['網格座標_R'] = pd.to_numeric(df['網格座標_R'], errors='coerce')
        df['網格座標_C'] = pd.to_numeric(df['網格座標_C'], errors='coerce')
        df.dropna(subset=['網格座標_R', '網格座標_C'], inplace=True)
        
        filtered_df = df[df['資料來源'] == source_filter].copy()
        
        required_cols = {'網格座標_R': 'R', '網格座標_C': 'C', 'MAE': 'MAE'}
        if not all(col in filtered_df.columns for col in required_cols.keys()):
            logger.warning(f"檔案 {os.path.basename(filepath)} 中缺少必要的欄位。")
            return pd.DataFrame()
            
        return filtered_df[list(required_cols.keys())].rename(columns=required_cols)

    except FileNotFoundError:
        logger.error(f"錯誤：找不到檔案 '{filepath}'。")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"讀取或處理檔案 '{filepath}' 時發生錯誤: {e}")
        return pd.DataFrame()

# ==============================================================================
# 主執行流程
# ==============================================================================
def main():
    logger = setup_logging()
    logger.info("===== 開始整合並繪製 MAE 地圖 =====")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    H, W = CONFIG["grid_shape"]
    mae_grid = np.full((H, W), np.nan)
    source_grid = np.full((H, W), -1, dtype=int)

    source_labels = [get_source_label(p) for p in CONFIG["input_excel_paths"]]
    
    for i, filepath in enumerate(CONFIG["input_excel_paths"]):
        logger.info(f"--- 正在處理來源 '{source_labels[i]}' ({os.path.basename(filepath)}) ---")
        stage4_df = load_data_from_excel(filepath, 'stage4_model')
        if stage4_df.empty: continue
        for _, row in stage4_df.iterrows():
            r, c, mae = int(row['R']), int(row['C']), row['MAE']
            if 0 <= r < H and 0 <= c < W:
                if np.isnan(mae_grid[r, c]) or mae < mae_grid[r, c]:
                    mae_grid[r, c] = mae
                    source_grid[r, c] = i
                    
    filler_label = "stage3_fallback"
    filler_source_index = len(source_labels)
    source_labels.append(filler_label)
    stage3_df = load_data_from_excel(CONFIG["filler_excel_path"], 'stage3_model')
    if not stage3_df.empty:
        for _, row in stage3_df.iterrows():
            r, c, mae = int(row['R']), int(row['C']), row['MAE']
            if 0 <= r < H and 0 <= c < W:
                if np.isnan(mae_grid[r, c]):
                    mae_grid[r, c] = mae
                    source_grid[r, c] = filler_source_index

    try:
        mapping_df = pd.read_excel(CONFIG["mapping_table_path"])
    except FileNotFoundError:
        logger.error(f"錯誤：找不到座標映射表 '{CONFIG['mapping_table_path']}'。無法繪圖。")
        return

    plot_df = pd.DataFrame({
        'R': np.arange(H * W) // W, 'C': np.arange(H * W) % W,
        'MAE': mae_grid.flatten(), 'source_idx': source_grid.flatten()
    })
    
    final_plot_df = pd.merge(plot_df, mapping_df, on=['R', 'C'], how='left')
    final_plot_df.dropna(subset=['lon', 'lat', 'MAE'], inplace=True)
    logger.info(f"資料整合完成，共 {len(final_plot_df)} 個有效網格點可供繪製。")

    if final_plot_df.empty:
        logger.error("沒有可供繪製的數據。")
        return

    # --- 【核心修改】準備外框顏色與圖例 ---
    # 1. 建立一個完整的顏色列表，用於「圖例」顯示
    outline_color_map = plt.cm.get_cmap('tab10', len(source_labels))
    colors_for_legend = [outline_color_map(i) for i in range(len(source_labels))]
    
    # 2. 根據這個顏色列表，預先產生圖例物件
    legend_patches = [mpatches.Patch(color=colors_for_legend[i], label=label) for i, label in enumerate(source_labels)]

    # 3. 建立一個實際用於「繪圖」的顏色列表，並將 Stage3 的顏色設定為 'none'
    colors_for_plotting = list(colors_for_legend) # 複製一份
    colors_for_plotting[filler_source_index] = 'none' # 將 Stage3 的外框設為無顏色
    
    # 4. 為每個點指定其外框顏色
    final_plot_df['outline_color'] = final_plot_df['source_idx'].apply(lambda idx: colors_for_plotting[int(idx)] if idx != -1 else 'none')
    # --- 修改結束 ---

    logger.info("--- 開始繪製最終地圖 ---")
    fig, ax = plt.subplots(figsize=(15, 12))
    plot_cfg = CONFIG["plot_config"]

    ax.scatter(
        x=final_plot_df['lon'],
        y=final_plot_df['lat'],
        facecolor=plot_cfg["fill_color"],
        s=plot_cfg["marker_size"],
        marker='s',
        edgecolor=final_plot_df['outline_color'],
        linewidth=plot_cfg["outline_linewidth"]
    )
    
    text_cfg = plot_cfg["text_label"]
    if text_cfg["enabled"]:
        for _, row in final_plot_df.iterrows():
            ax.text(
                x=row['lon'], y=row['lat'], s=f"{row['MAE']:.0f}",
                color=text_cfg["color"], fontsize=text_cfg["fontsize"],
                ha='center', va='center'
            )

    # 使用預先產生的圖例物件
    ax.legend(handles=legend_patches, title='Data Source (Outline Color)', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Combined MAE Distribution Map by Source", fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)

    output_path = os.path.join(CONFIG["output_dir"], CONFIG["output_filename"])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"地圖已成功儲存至: {output_path}")
    except Exception as e:
        logger.error(f"儲存地圖失敗: {e}")
    finally:
        plt.close(fig)

    logger.info("===== 所有任務執行完畢 =====")

if __name__ == '__main__':
    main()
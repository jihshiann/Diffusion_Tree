import os
import glob
import re
import pandas as pd
import numpy as np
from functools import reduce
import math
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定義檢查並刪除高比例零值的函數
def remove_high_zero_columns(df, threshold=0.05):
    """刪除零值比例高於閾值的欄位"""
    # 計算每欄中零值的比例
    zero_ratio = (df == 0).mean(axis=0)
    # 找出零值比例超過閾值的欄位
    columns_to_remove = zero_ratio[zero_ratio > threshold].index
    # 記錄刪除的欄位數量
    if len(columns_to_remove) > 0:
        logger.info(f"刪除 {len(columns_to_remove)} 個零值比例過高的欄位")
    # 刪除這些欄位
    df.drop(columns=columns_to_remove, inplace=True)
    return df

# 定義針對每個座標欄位，將0或空值改成該座標欄位前一筆的值(前一小時)
def fill_zero_with_previous(df):
    """將零值或空值用前一筆資料填補"""
    # 對除了「時間」之外的所有欄位進行處理
    cols = df.columns.drop("時間") if "時間" in df.columns else df.columns
    for col in cols:
        # 將0值替換成NA
        df[col] = df[col].replace(0, pd.NA)
        # 用前一筆資料填補
        df[col] = df[col].ffill()
        # 若仍有缺失值（例如第一筆），以0填補
        df[col] = df[col].fillna(0)
        # 轉為整數，但先檢查是否可以轉換
        try:
            df[col] = df[col].astype(int)
        except (ValueError, TypeError):
            # 如果不能轉為整數（例如非數值欄位），保持原樣
            pass
    return df

# 修正座標欄位名稱函數
def fix_coordinate_column_names(df):
    """修正如 '(121.525, 25.105)_x' 等格式的座標欄位名稱為 '(121.525, 25.105)'"""
    new_columns = {}
    pattern = re.compile(r"(\(.*?\))(_.*)?")
    for col in df.columns:
        match = pattern.fullmatch(col)
        if match:
            correct_name = match.group(1)
            if correct_name != col:
                new_columns[col] = correct_name
    df.rename(columns=new_columns, inplace=True)
    return df

# 移除重複欄位，只保留第一個
def remove_duplicate_columns_keep_first(df):
    """只保留第一次出現的欄位，移除重複名稱的欄位"""
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicated_cols:
        logger.info(f"移除 {len(duplicated_cols)} 個重複欄位: {duplicated_cols}")
    return df.loc[:, ~df.columns.duplicated()]

# 主程式開始
def main():
    try:
        # 讀取外部天氣資料 (External.txt)
        external_path = r"C:\thesis\code\External.txt"
        logger.info(f"開始讀取外部天氣資料: {external_path}")
        
        try:
            ext_df = pd.read_csv(external_path)
            logger.info(f"成功讀取外部資料，共 {len(ext_df)} 筆記錄")
        except Exception as e:
            logger.error(f"讀取外部資料失敗: {e}")
            return
        
        # 處理 hoilday 欄位 (轉換為數值型資料 0 或 1)
        # 修正：直接檢查字串是否為 'True'，避免大小寫問題
        ext_df['hoilday'] = ext_df['hoilday'].apply(
            lambda x: 1 if str(x).lower() == 'true' else (0 if str(x).lower() == 'false' else x)
        )
        
        # 確保 hoilday 欄位為整數
        try:
            ext_df['hoilday'] = pd.to_numeric(ext_df['hoilday'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            logger.warning(f"轉換 hoilday 欄位時出錯: {e}")
        
        # 處理降水量欄位
        if '降水量' in ext_df.columns:
            # 將 "降水量" 欄位中的 'T' 轉換為 0.1 (微量)
            ext_df['降水量'] = ext_df['降水量'].replace('T', 0.1)
            # 確保降水量為數值
            ext_df['降水量'] = pd.to_numeric(ext_df['降水量'], errors='coerce').fillna(0)
        
        # 檢查 ext_df，針對每個時間資料（除「時間」欄之外），若有缺失值則直接拿掉該列
        na_rows_before = len(ext_df)
        ext_df = ext_df.dropna()
        na_rows_after = len(ext_df)
        
        if na_rows_before > na_rows_after:
            logger.info(f"移除了 {na_rows_before - na_rows_after} 筆含有缺失值的資料列")
        
        # 確保時間欄位為 datetime 格式
        if '時間' in ext_df.columns:
            try:
                ext_df['時間'] = pd.to_datetime(ext_df['時間'])
            except Exception as e:
                logger.error(f"轉換時間欄位時出錯: {e}")
                return
        else:
            logger.error("外部資料中找不到「時間」欄位")
            return
        
        # 流量資料所在資料夾
        folder_path = r"C:\thesis\code\Taipei_CF"
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            logger.error(f"在 {folder_path} 中找不到任何 CSV 檔案")
            return
        
        logger.info(f"找到 {len(csv_files)} 個流量數據檔案")
        
        # 用來儲存處理後的各檔案 DataFrame
        dfs = []
        
        # 逐一讀取並處理每個流量檔案
        for file_idx, file_path in enumerate(csv_files, 1):
            try:
                logger.info(f"處理檔案 {file_idx}/{len(csv_files)}: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path)
                
                # 假設第一欄即為時間，轉換為 datetime 格式，並重新命名為 "時間"
                time_col = df.columns[0]
                df[time_col] = pd.to_datetime(df[time_col])
                df.rename(columns={time_col: "時間"}, inplace=True)
                
                # 針對每個座標欄位，移除零值比例過高的欄位（columns）
                df_cols_before = df.shape[1]
                df = remove_high_zero_columns(df)
                df_cols_after = df.shape[1]
                
                if df_cols_before > df_cols_after:
                    logger.info(f"移除了 {df_cols_before - df_cols_after} 個零值比例過高的欄位")
                
                # 針對每個時間列，移除零值比例過高的資料列（rows）
                rows_before = df.shape[0]
                zero_ratio_rows = (df.drop(columns=['時間']) == 0).mean(axis=1)
                df = df[zero_ratio_rows <= 0.1].reset_index(drop=True)
                rows_after = df.shape[0]
                
                if rows_before > rows_after:
                    logger.info(f"移除了 {rows_before - rows_after} 筆零值比例過高的資料列")
                
                # 針對每個座標欄位，將0或空值改成該座標欄位前一筆的值(前一小時)
                df = fill_zero_with_previous(df)
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"處理檔案 {file_path} 時出錯: {e}")
                continue
        
        # 檢查是否有處理後的資料框
        if not dfs:
            logger.error("沒有成功處理任何檔案，無法繼續")
            return
        
        # 先依據「時間」欄位合併所有流量檔案（以 outer join 保留所有時間）
        logger.info("開始合併所有流量檔案...")
        try:
            merged_flow_df = reduce(lambda left, right: pd.merge(left, right, on="時間", how="outer"), dfs)
            logger.info(f"合併後的資料有 {merged_flow_df.shape[0]} 筆記錄，{merged_flow_df.shape[1]} 個欄位")
        except Exception as e:
            logger.error(f"合併流量檔案時出錯: {e}")
            return
        
        # 以外部天氣資料 ext_df 為基準，僅保留外部資料中存在的時間（inner join）
        logger.info("將流量資料與外部天氣資料合併...")
        try:
            rows_before = merged_flow_df.shape[0]
            final_df = pd.merge(merged_flow_df, ext_df, on="時間", how="inner")
            rows_after = final_df.shape[0]
            
            logger.info(f"合併後保留 {rows_after} 筆記錄，捨棄了 {rows_before - rows_after} 筆記錄")
        except Exception as e:
            logger.error(f"合併外部資料時出錯: {e}")
            return
        
        # 修正座標欄位名稱
        final_df = fix_coordinate_column_names(final_df)
        
        # 檢查座標是否有重複，重複的話只保留一個
        final_df = remove_duplicate_columns_keep_first(final_df)
        
        # 拿掉2020以後的資料
        rows_before = final_df.shape[0]
        final_df = final_df[final_df['時間'].dt.year < 2020].reset_index(drop=True)
        rows_after = final_df.shape[0]
        
        if rows_before > rows_after:
            logger.info(f"移除了 {rows_before - rows_after} 筆2020年以後的資料")
        
        # 處理座標矩陣
        logger.info("處理座標矩陣...")
        
        # 1. 提取所有座標欄位 (假設座標欄位以 "(" 開頭且以 ")" 結尾)
        coord_columns = [col for col in final_df.columns if col.startswith("(") and col.endswith(")")]
        logger.info(f"找到 {len(coord_columns)} 個座標欄位")
        
        # 2. 建立 mapping，由 (lon, lat) tuple 對應到原始欄位名稱
        coord_dict = {}
        valid_coords = 0
        for col in coord_columns:
            try:
                content = col.strip("()")
                lon_str, lat_str = content.split(",")
                lon = float(lon_str.strip())
                lat = float(lat_str.strip())
                coord_dict[(lon, lat)] = col
                valid_coords += 1
            except Exception as e:
                logger.warning(f"處理座標欄位 '{col}' 時出錯: {e}")
                continue
        
        logger.info(f"成功解析 {valid_coords} 個有效座標")
        
        # 3. 取得所有唯一的經度與緯度，並排序
        unique_lons = sorted(set(lon for lon, lat in coord_dict.keys()))
        unique_lats = sorted(set(lat for lon, lat in coord_dict.keys()), reverse=True)  # 緯度由大到小排序
        
        logger.info(f"有 {len(unique_lons)} 個唯一經度值和 {len(unique_lats)} 個唯一緯度值")
        
        # 4. 根據總座標數量決定最大的正方形邊長
        total_coords = len(coord_dict)
        n = int(math.floor(math.sqrt(total_coords)))  # n 為邊長，例如 509 -> floor(sqrt(509)) = 22
        n = 23
        
        logger.info(f"總共有 {total_coords} 個座標，將選取 {n}x{n}={n*n} 個座標")
        
        # 5. 定義取中間 n 個元素的函數（若數量不足則全部保留）
        def select_center(lst, n):
            if len(lst) <= n:
                return lst
            start = (len(lst) - n) // 2
            return lst[start:start+n]
        
        # 選擇中間的經緯度
        selected_lons = select_center(unique_lons, n)
        selected_lats = select_center(unique_lats, n)
        
        logger.info(f"選取了 {len(selected_lons)} 個經度和 {len(selected_lats)} 個緯度")
        
        # 6. 確認我們選擇的經緯度組合是否存在於原始數據中
        # 創建一個集合來存儲實際存在的座標
        available_coords = set(coord_dict.keys())
        
        # 建立允許保留的座標集合 - 這些是理論上的 n*n 網格
        allowed_coords = set()
        for lat in selected_lats:
            for lon in selected_lons:
                allowed_coords.add((lon, lat))
        
        # 計算實際可用的座標（理論網格點與實際座標的交集）
        actual_allowed_coords = allowed_coords.intersection(available_coords)
        logger.info(f"理論網格點: {len(allowed_coords)}，實際存在的網格點: {len(actual_allowed_coords)}")
        
        # 如果實際可用座標少於預期，可以嘗試擴大選擇範圍
        if len(actual_allowed_coords) < n*n:
            logger.warning(f"實際可用座標({len(actual_allowed_coords)})少於預期({n*n})，嘗試擴大選擇範圍")
            
            # 計算需要擴展的經緯度數量
            missing = n*n - len(actual_allowed_coords)
            extend_count = math.ceil(math.sqrt(missing))
            
            # 擴展經度和緯度的選擇
            if len(unique_lons) > len(selected_lons):
                extended_lons = select_center(unique_lons, len(selected_lons) + extend_count)
                logger.info(f"擴展經度選擇範圍至 {len(extended_lons)} 個")
            else:
                extended_lons = selected_lons
            
            if len(unique_lats) > len(selected_lats):
                extended_lats = select_center(unique_lats, len(selected_lats) + extend_count)
                logger.info(f"擴展緯度選擇範圍至 {len(extended_lats)} 個")
            else:
                extended_lats = selected_lats
            
            # 重新計算允許的座標集合
            allowed_coords = set()
            for lat in extended_lats:
                for lon in extended_lons:
                    allowed_coords.add((lon, lat))
            
            actual_allowed_coords = allowed_coords.intersection(available_coords)
            logger.info(f"擴展後實際可用座標: {len(actual_allowed_coords)}")
        
        # 7. 刪除不屬於選定範圍的座標欄位
        cols_before = final_df.shape[1]
        dropped_coords = []
        
        for (lon, lat), col in list(coord_dict.items()):
            if (lon, lat) not in actual_allowed_coords:
                if col in final_df.columns:
                    final_df.drop(columns=col, inplace=True)
                    dropped_coords.append(col)
        
        cols_after = final_df.shape[1]
        logger.info(f"移除了 {cols_before - cols_after} 個不在選定範圍內的座標欄位")
        
        # 8. 儲存保留的座標，用於檢查
        retained_coords = [(lon, lat) for (lon, lat) in coord_dict.keys() if (lon, lat) in actual_allowed_coords]
        logger.info(f"保留的座標數量: {len(retained_coords)}")
        
        # 9. 依照緯度（由大到小）與經度（由小到大）重新排序剩下的座標欄位
        new_coord_order = []
        for lat in sorted(set(lat for lon, lat in retained_coords), reverse=True):
            for lon in sorted(set(lon for lon, lat in retained_coords if lat == lat)):
                key = (lon, lat)
                if key in coord_dict and coord_dict[key] in final_df.columns:
                    new_coord_order.append(coord_dict[key])
        
        logger.info(f"重新排序後的座標欄位數量: {len(new_coord_order)}")
        
        # 檢查最終結果是否達到預期
        if len(new_coord_order) < n*n:
            logger.warning(f"警告: 最終保留的座標數量({len(new_coord_order)})小於預期的({n*n})")
        
        # 10. 重新排列 final_df 的欄位：將「時間」放最前面，其次為排序後的座標欄位，再接著其他非座標欄位
        non_coord_columns = [col for col in final_df.columns 
                           if not (col.startswith("(") and col.endswith(")")) and col != "時間"]
        
        final_df = final_df[["時間"] + new_coord_order + non_coord_columns]
        
        # 新增年、月、日、時欄位
        logger.info("新增時間相關欄位...")
        final_df['年'] = final_df['時間'].dt.year
        final_df['月'] = final_df['時間'].dt.month
        final_df['日'] = final_df['時間'].dt.day
        final_df['時'] = final_df['時間'].dt.hour
        
        # 依據時間排序並重設索引
        final_df.sort_values("時間", inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        # 針對流量資料進行最後處理：填補缺失值
        flow_cols = final_df.columns.drop(["時間", "年", "月", "日", "時"]) if all(x in final_df.columns for x in ["時間", "年", "月", "日", "時"]) else final_df.columns
        
        nulls_before = final_df[flow_cols].isnull().sum().sum()
        
        for col in flow_cols:
            final_df[col] = final_df[col].ffill()  # 將 NaN 的位置用「前一筆資料」填補
            
            # 若最前面仍是 NaN，就用 0 填補，並嘗試轉為整數
            try:
                final_df[col] = final_df[col].fillna(0)
                
                # 檢查是否可以轉為整數
                if pd.api.types.is_numeric_dtype(final_df[col]):
                    final_df[col] = final_df[col].astype(int)
            except Exception as e:
                logger.warning(f"處理欄位 '{col}' 時出錯: {e}")
        
        nulls_after = final_df[flow_cols].isnull().sum().sum()
        logger.info(f"填補了 {nulls_before - nulls_after} 個缺失值")
        
        # 輸出最終合併結果至單一檔案
        output_file = os.path.join(folder_path, "all_merged.csv")
        logger.info(f"儲存結果至: {output_file}")
        
        try:
            final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            logger.info(f"成功將資料儲存至: {output_file}")
            logger.info(f"最終資料大小: {final_df.shape[0]} 筆記錄 x {final_df.shape[1]} 個欄位")
        except Exception as e:
            logger.error(f"儲存檔案時出錯: {e}")
    
    except Exception as e:
        logger.error(f"程式執行過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
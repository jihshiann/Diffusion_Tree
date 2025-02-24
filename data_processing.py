import os
import re
import glob
import pandas as pd
from functools import reduce

# TODO: 存檔時依照座標分數個資料夾存


# 定義檢查並刪除高比例零值的函數
def remove_high_zero_columns(df, threshold=0.05):
    # 計算每列中零值的比例
    zero_ratio = (df == 0).mean(axis=0)
    # 找出零值比例超過閾值的列
    columns_to_remove = zero_ratio[zero_ratio > threshold].index
    # 刪除這些列
    df.drop(columns=columns_to_remove, inplace=True)
    return df

# 讀取外部天氣資料 (External.txt)
external_path = r"C:\thesis\code\External.txt"
ext_df = pd.read_csv(external_path)

# 處理 hoilday 欄位 (轉換為數值型資料 0 或 1)
ext_df['hoilday'] = ext_df['hoilday'].apply(lambda x: 1 if x == 'True' else (0 if x == 'False' else x))
ext_df['hoilday'] = pd.to_numeric(ext_df['hoilday'], errors='coerce').fillna(0).astype(int)

# 將 "降水量" 欄位轉成數值
ext_df['降水量'] = ext_df['降水量'].replace('T', 0.1)
ext_df['降水量'] = pd.to_numeric(ext_df['降水量'])

ext_df = ext_df.fillna(0)

# 假設外部檔案第一欄標題為「時間」，轉換為 datetime 格式
ext_df['時間'] = pd.to_datetime(ext_df['時間'])

# 流量資料所在資料夾（此處假設檔案格式為 CSV，如果是 Excel 檔請使用 pd.read_excel）
folder_path = r"C:\thesis\code\Taipei_5x5"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 用來儲存處理後的各檔案 DataFrame
dfs = []

# 逐一讀取並處理每個流量檔案
for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    # 假設第一欄即為時間，取出第一欄名稱，轉換為 datetime 格式，並重新命名為 "時間"
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df.rename(columns={time_col: "時間"}, inplace=True)
    
    # 針對每個座標列，移除零值比例過高的列
    df = remove_high_zero_columns(df)

    # 針對每個時間欄，移除零值比例過高的欄
    zero_ratio_rows = (df.drop(columns=['時間']) == 0).mean(axis=1)
    df = df[zero_ratio_rows <= 0.1].reset_index(drop=True)

    # 針對每個座標列，將0的資料改為該座標列的平均值
    numeric_cols = df.columns.drop('時間')
    col_means = df[numeric_cols].replace(0, pd.NA).mean()
    for col in numeric_cols:
        df[col] = df[col].replace(0, col_means[col]).round(0).astype(int)

    dfs.append(df)

# 先依據「時間」欄位合併所有流量檔案（以 outer join 保留所有時間）
if dfs:
    merged_flow_df = reduce(lambda left, right: pd.merge(left, right, on="時間", how="outer"), dfs)
else:
    merged_flow_df = pd.DataFrame()

# inner join
final_df = pd.merge(merged_flow_df, ext_df, on="時間", how="inner")

# 新增年、月、日、時欄位
final_df['年'] = final_df['時間'].dt.year
final_df['月'] = final_df['時間'].dt.month
final_df['日'] = final_df['時間'].dt.day
final_df['時'] = final_df['時間'].dt.hour

# 依據時間排序並重設索引
final_df.sort_values("時間", inplace=True)
final_df.reset_index(drop=True, inplace=True)

def fix_coordinate_column_names(df):
    # 建立新的欄位映射字典
    new_columns = {}
    # 正規表達式：第一組捕捉正確格式的座標，第二組捕捉可能存在的後綴
    pattern = re.compile(r"(\(.*?\))(_.*)?")
    for col in df.columns:
        # 嘗試完全匹配欄位名稱
        match = pattern.fullmatch(col)
        if match:
            # 使用第一組 (正確的座標部分) 作為新的名稱
            correct_name = match.group(1)
            # 若新名稱與原名稱不同，則記錄映射關係
            if correct_name != col:
                new_columns[col] = correct_name
    # 重新命名 DataFrame 的欄位
    df.rename(columns=new_columns, inplace=True)
    return df

# 在輸出 final_df 之前先修正座標欄位名稱
final_df = fix_coordinate_column_names(final_df)

# TODO: 檢查座標是否有格式不對的

# TODO: 拿掉2020以後的資料


# 輸出最終合併結果至單一檔案
output_file = os.path.join(folder_path, "all_merged_5X5.csv")
final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"所有檔案合併後已儲存至: {output_file}")

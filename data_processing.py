import os
import glob
import re
import pandas as pd
from functools import reduce

# 定義檢查並刪除高比例零值的函數
def remove_high_zero_columns(df, threshold=0.05):
    # 計算每欄中零值的比例
    zero_ratio = (df == 0).mean(axis=0)
    # 找出零值比例超過閾值的欄位
    columns_to_remove = zero_ratio[zero_ratio > threshold].index
    # 刪除這些欄位
    df.drop(columns=columns_to_remove, inplace=True)
    return df

# 定義針對每個座標欄位，將0或空值改成該座標欄位前一筆的值(前一小時)
def fill_zero_with_previous(df):
    # 對除了「時間」之外的所有欄位進行處理
    cols = df.columns.drop("時間")
    for col in cols:
        # 將0值替換成NA
        df[col] = df[col].replace(0, pd.NA)
        # 用前一筆資料填補
        df[col] = df[col].ffill()
        # 若仍有缺失值（例如第一筆），以0填補
        df[col] = df[col].fillna(0)
        # 轉為整數
        df[col] = df[col].astype(int)
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

# 檢查 ext_df，針對每個時間資料（除「時間」欄之外），若有缺失值則直接拿掉該列
ext_df = ext_df.dropna()

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
    
    # 假設第一欄即為時間，轉換為 datetime 格式，並重新命名為 "時間"
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df.rename(columns={time_col: "時間"}, inplace=True)
    
    # 針對每個座標欄位，移除零值比例過高的欄位（columns）
    df = remove_high_zero_columns(df)
    
    # 針對每個時間列，移除零值比例過高的資料列（rows）
    zero_ratio_rows = (df.drop(columns=['時間']) == 0).mean(axis=1)
    df = df[zero_ratio_rows <= 0.1].reset_index(drop=True)
    
    # 針對每個座標欄位，將0或空值改成該座標欄位前一筆的值(前一小時)
    dfs.append(df)

# 先依據「時間」欄位合併所有流量檔案（以 outer join 保留所有時間）
if dfs:
    merged_flow_df = reduce(lambda left, right: pd.merge(left, right, on="時間", how="outer"), dfs)
else:
    merged_flow_df = pd.DataFrame()

# 以外部天氣資料 ext_df 為基準，僅保留外部資料中存在的時間（inner join）
final_df = pd.merge(merged_flow_df, ext_df, on="時間", how="inner")

# 在輸出前先檢查並修正座標欄位名稱：若有誤格式如 "(121.525, 25.105)_x" 則修正為 "(121.525, 25.105)"
def fix_coordinate_column_names(df):
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

final_df = fix_coordinate_column_names(final_df)


# TODO: 檢查座標是否有重複，重複的話只保留一個
def remove_duplicate_columns_keep_first(df):
    """
    只保留第一次出現的欄位，對於重複出現的欄位名稱，只保留第一個，
    使用 df.loc[:, ~df.columns.duplicated()] 可直接解決此問題。
    """
    return df.loc[:, ~df.columns.duplicated()]

final_df = remove_duplicate_columns_keep_first(final_df)


# 拿掉2020以後的資料
final_df = final_df[final_df['時間'].dt.year < 2020].reset_index(drop=True)

# 新增年、月、日、時欄位
final_df['年'] = final_df['時間'].dt.year
final_df['月'] = final_df['時間'].dt.month
final_df['日'] = final_df['時間'].dt.day
final_df['時'] = final_df['時間'].dt.hour

# 依據時間排序並重設索引
final_df.sort_values("時間", inplace=True)
final_df.reset_index(drop=True, inplace=True)

flow_cols = final_df.columns.drop(["時間", "年", "月", "日", "時"])  # 排除時間和年、月、日、時
for col in flow_cols:
    final_df[col] = final_df[col].ffill()  # 將 NaN 的位置用「前一筆資料」填補
    final_df[col] = final_df[col].fillna(0).astype(int)  # 若最前面仍是 NaN，就用 0


# 輸出最終合併結果至單一檔案
output_file = os.path.join(folder_path, "all_merged_5X5.csv")
final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"所有檔案合併後已儲存至: {output_file}")

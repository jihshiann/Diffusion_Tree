import pandas as pd

# --- 1. 設定檔案路徑 ---
# 注意：讀取和儲存的路徑現在是同一個檔案
path_events = r"C:\thesis\code\Taipei_CF\ArenaEvents.xlsx"
path_all_merged_original = r"C:\thesis\code\Taipei_CF\all_merged.csv"


try:
    # --- 2. 讀取兩個檔案 ---
    print("正在讀取檔案...")
    df_events = pd.read_excel(path_events)
    df_all = pd.read_csv(path_all_merged_original)
    print("檔案讀取完成。")

    # --- 3. 準備 ArenaEvents 的日期資料 ---
    # 選取'年', '月', '日'三欄，並移除重複的日期組合
    event_dates = df_events[['年', '月', '日']].drop_duplicates().copy()
    
    # 新增一個標記欄位，用於合併後識別
    event_dates['is_event_day'] = 1

    # --- 4. 合併兩個 DataFrame ---
    # 使用左合併 (left merge)，以 df_all 為主體
    print("正在比對日期並合併資料...")
    df_final = pd.merge(df_all, event_dates, on=['年', '月', '日'], how='left')

    # --- 5. 建立新的 'ArenaEvents' 欄位 ---
    # 將合併後產生的輔助欄位 'is_event_day' 的 NaN 填補為 0
    df_final['ArenaEvents'] = df_final['is_event_day'].fillna(0)
    
    # 將欄位型態轉換為整數 (integer)
    df_final['ArenaEvents'] = df_final['ArenaEvents'].astype(int)
    
    # 移除輔助用的 'is_event_day' 欄位
    df_final = df_final.drop(columns=['is_event_day'])

    print("新欄位 'ArenaEvents' 已成功建立。")

    # --- 6. 儲存結果並覆蓋原檔案 ---
    # **********************************************************
    # * 注意：這裡會將結果直接寫回 path_all_merged_original *
    # **********************************************************
    # index=False 表示儲存時不要把 DataFrame 的索引也存進去
    # encoding='utf-8-sig' 確保中文能正確寫入與讀取
    print(f"正在儲存結果並覆蓋原始檔案: {path_all_merged_original}")
    df_final.to_csv(path_all_merged_original, index=False, encoding='utf-8-sig')
    print(f"處理完成！原始檔案已成功更新。")
    
    # --- 7. (可選) 顯示處理後的前幾筆資料預覽 ---
    print("\n更新後資料預覽 (前 5 筆):")
    print(df_final.head())
    
    print("\n欄位 'ArenaEvents' 的數值分佈:")
    print(df_final['ArenaEvents'].value_counts())


except FileNotFoundError as e:
    print(f"錯誤：找不到檔案！請檢查路徑是否正確。\n{e}")
except Exception as e:
    print(f"發生未預期的錯誤：\n{e}")
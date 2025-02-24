import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os

# 設定 Matplotlib 字體，確保 SHAP 可視化時能正常顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取 CSV 檔案
data_path = r"C:\thesis\code\Taipei_5x5\all_merged_5X5.csv"
df = pd.read_csv(data_path)

# 設定輸出目錄
result_dir = r"C:\thesis\code\result"
os.makedirs(result_dir, exist_ok=True)  # 若資料夾不存在則創建


# 處理角度變數
df['sin_max_gust_direction'] = np.sin(np.deg2rad(df['最大陣風風向']))  # 角度轉為sin值
df['cos_max_gust_direction'] = np.cos(np.deg2rad(df['最大陣風風向']))  # 角度轉為cos值
df['sin_wind_direction'] = np.sin(np.deg2rad(df['風向']))  # 角度轉為sin值
df['cos_wind_direction'] = np.cos(np.deg2rad(df['風向']))  # 角度轉為cos值

# 建立特徵名稱翻譯對應表（僅適用於決策樹）
feature_mapping = {
    '測站氣壓': 'Station_Pressure',
    '海平面氣壓': 'Sea_Level_Pressure',
    '氣溫': 'Temperature',
    '露點溫度': 'Dew_Point',
    '相對溼度': 'Relative_Humidity',
    '風速': 'Wind_Speed',
    '風向': 'Wind_Direction',
    '最大陣風': 'Max_Gust',
    '最大陣風風向': 'Max_Gust_Direction',
    '降水量': 'Precipitation',
    '降水時數': 'Precipitation_Hours',
    '日照時數': 'Sunshine_Hours',
    '全天空日射量': 'Global_Radiation',
    '能見度': 'Visibility',
    '紫外線指數': 'UV_Index',
    '總雲量': 'Total_Cloud_Cover',
    'hoilday': 'Holiday',
    'weekday': 'Weekday',
    '年': 'Year',
    '月': 'Month',
    '日': 'Day',
    '時': 'Hour'
}

# 保留原始 DataFrame
df_original = df.copy()

# 提取座標欄位
target_columns = [col for col in df.columns if '(' in col and ')' in col][:3]

# 顯示提取出來的所有座標點
print("所有座標點：")
print(target_columns)

# 替換 DataFrame 欄位名稱為英文
df_tree = df.rename(columns=feature_mapping)

# 設定 X (特徵) 和 y (目標)
X = df_original[list(feature_mapping.keys())] 
y = df[target_columns]

# 切分訓練集與測試集
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tree = X_train.rename(columns=feature_mapping)  
X_test_tree = X_test.rename(columns=feature_mapping) 

# 定義類別型特徵
cat_features = ['Holiday', 'Weekday', 'Month', 'Day', 'Hour']

# 用來儲存每個目標的預測結果
predictions = {}

# 逐一為每個座標點訓練模型
for target in target_columns:
    print(f"訓練與預測 {target} 的決策樹...")

    train_data = lgb.Dataset(X_train_tree, label=y_train[target], categorical_feature=cat_features)
    test_data = lgb.Dataset(X_test_tree, label=y_test[target], reference=train_data, categorical_feature=cat_features)

    # 設定 LightGBM 模型參數
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,#2^n-1
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'seed': 42
    }

    # 記錄 RMSE 變化
    evals_result = {}

    # 訓練模型
    lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=300,  # 設定最大迭代輪數
    valid_sets=[test_data],  # 使用測試集作為驗證集
    valid_names=["valid_0"],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True),  # 若沒有改善則停止
               lgb.record_evaluation(evals_result),  # 記錄評估結果
               lgb.log_evaluation(10)  
    ])

    # 預測
    y_pred = lgb_model.predict(X_test_tree, num_iteration=lgb_model.best_iteration)

    # 儲存預測結果
    predictions[target] = y_pred
    print(f"LightGBM 總樹數: {lgb_model.num_trees()}")

    # 繪製學習曲線
    if "valid_0" in evals_result and "rmse" in evals_result["valid_0"]:
        plt.figure(figsize=(8, 5))
        plt.plot(evals_result['valid_0']['rmse'], label="Validation RMSE", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.title(f"Learning Curve ({target})")
        plt.legend()
        learning_curve_path = os.path.join(result_dir, f"learning_curve_{target.replace(',', '_').replace(' ', '')}.png")
        plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"學習曲線已儲存至: {learning_curve_path}")
    else:
        print(f"無法繪製 {target} 的學習曲線，因為 evals_result 沒有數據。")

    # ------------------------
    # 1. 繪製最後決策樹
    # plt.figure(figsize=(20, 10))
    # lgb.plot_tree(lgb_model, tree_index=lgb_model.best_iteration-1, show_info=['split_gain'])
    # plt.title(f"Best Decision Tree for {target}")
    # tree_plot_path = os.path.join(result_dir, f"best_tree_{target.replace(',', '_').replace(' ', '')}.png")
    # plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    # plt.close()
    # print(f"最佳決策樹圖已儲存至: {tree_plot_path}")

    # 繪製最佳決策樹
    # 取得模型結構
    model_dict = lgb_model.dump_model()
    tree_info = model_dict["tree_info"]
    # 定義遞迴函數以計算每棵樹的總 split_gain
    def get_total_gain(node):
        gain = node.get("split_gain", 0)
        if "left_child" in node:
            gain += get_total_gain(node["left_child"])
        if "right_child" in node:
            gain += get_total_gain(node["right_child"])
        return gain

    tree_gains = []
    for tree in tree_info:
        total_gain = get_total_gain(tree["tree_structure"])
        tree_gains.append(total_gain)

    best_tree_index = np.argmax(tree_gains)

    plt.figure(figsize=(30, 18))
    lgb.plot_tree(lgb_model, tree_index=best_tree_index, show_info=['split_gain'])
    plt.title(f"Best Decision Tree for {target}")
    tree_plot_path = os.path.join(result_dir, f"best_tree_{target.replace(',', '_').replace(' ', '')}.png")
    plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"最佳決策樹圖已儲存至: {tree_plot_path}")


    # ------------------------
    # 2. 使用 SHAP 顯示特徵的重要性分析
    explainer = shap.TreeExplainer(lgb_model) 
    shap_values = explainer.shap_values(X_test)
    # SHAP Summary Plot 
    shap_summary_path = os.path.join(result_dir, f"shap_summary_{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP 特徵重要性圖已儲存至: {shap_summary_path}")

    # ------------------------
    # 3. SHAP 特徵影響條形圖
    shap_bar_path = os.path.join(result_dir, f"shap_bar_{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP 特徵影響條形圖已儲存至: {shap_bar_path}")





#%%
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import time

# ---------------------------
# 設定與數據讀取
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = r"C:\thesis\code\Taipei_CF\all_merged.csv"
df = pd.read_csv(data_path)
result_dir = r"C:\thesis\code\result_lgb"
os.makedirs(result_dir, exist_ok=True)
# 建立子目錄
sub_dirs = ["learning_curve", "tree", "shap_summary", "shap_bar", "model", "group_tree"]
for sub in sub_dirs:
    os.makedirs(os.path.join(result_dir, sub), exist_ok=True)

# 處理角度變數
for col in ['最大陣風風向', '風向']:
    df[f'sin_{col}'] = np.sin(np.deg2rad(df[col]))
    df[f'cos_{col}'] = np.cos(np.deg2rad(df[col]))

# 建立特徵名稱翻譯對應表
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
    #'年': 'Year',
    '月': 'Month',
    '日': 'Day',
    '時': 'Hour'
}
reverse_mapping = {v: k for k, v in feature_mapping.items()}
rule_statistics = {} # 用於存儲規則及其累計分數
df_original = df.copy()

# 提取座標欄位（假設格式為 "(經度, 緯度)"）
target_columns = [col for col in df.columns if '(' in col and ')' in col]
# 確保 target_columns 中的座標是唯一的，如果原始數據中可能有重複
target_columns = sorted(list(set(target_columns)))
print("所有座標點：", target_columns)

# 替換 DataFrame 欄位名稱為英文供 LightGBM 使用
df_tree = df.rename(columns=feature_mapping)

# 定義 X 與 y
X = df_original[list(feature_mapping.keys())]
y = df_original[target_columns]

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tree = X_train.rename(columns=feature_mapping)
X_test_tree = X_test.rename(columns=feature_mapping)

cat_features = ['Holiday']

# 解析座標字串函式
def parse_coord_string(coord_str):
    coord_str = coord_str.strip("() ")
    lon_str, lat_str = coord_str.split(",")
    return float(lon_str), float(lat_str)

# ---------------------------
# 定義輔助函式

def get_feature_gain_vector(node, vector):
    if "split_feature" in node:
        feat_index = node["split_feature"]
        gain = node.get("split_gain", 0)
        vector[feat_index] += gain
    if "left_child" in node:
        get_feature_gain_vector(node["left_child"], vector)
    if "right_child" in node:
        get_feature_gain_vector(node["right_child"], vector)
    return vector

def get_breadth_first_path(tree_structure):
    """以廣度優先順序遍歷樹，返回所有節點的 (split_feature, threshold) 規則，閾值四捨五入到小數點第一位"""
    path = []
    q = deque([tree_structure])
    while q:
        node = q.popleft()
        if "split_feature" in node:
            feature_idx = node.get("split_feature")
            threshold = node.get("threshold")
            # 對連續特徵的閾值四捨五入到小數點第一位
            feature_name = list(X_train_tree.columns)[feature_idx] if feature_idx < len(X_train_tree.columns) else str(feature_idx)
            if feature_name not in cat_features:
                try:
                    threshold = round(float(threshold), 1)
                except (TypeError, ValueError):
                    pass  # 若閾值無法轉為浮點數（例如類別特徵），保持不變
            rule = (feature_idx, threshold)
            path.append(rule)
            if "left_child" in node:
                q.append(node["left_child"])
            if "right_child" in node:
                q.append(node["right_child"])
    return path

def get_decision_path(tree_structure, max_depth=3):
    """以深度優先方式提取決策樹的前 max_depth 層規則路徑，包含特徵和閾值"""
    def traverse(node, current_path, depth):
        if depth > max_depth or "split_feature" not in node:
            return
        feature_idx = node["split_feature"]
        threshold = node.get("threshold")
        feature_name = list(X_train_tree.columns)[feature_idx] if feature_idx < len(X_train_tree.columns) else str(feature_idx)
        # 根據特徵類型決定比較運算符
        if feature_name in cat_features:
            rule = f"{feature_name} == {threshold}"
        else:
            rule = f"{feature_name} <= {threshold}"
        current_path.append(rule)
        if "left_child" in node:
            traverse(node["left_child"], current_path.copy(), depth + 1)
        if "right_child" in node:
            traverse(node["right_child"], current_path.copy(), depth + 1)
        if depth == max_depth:
            paths.append(current_path[:])

    paths = []
    traverse(tree_structure, [], 1)
    # 返回第一條路徑（或可根據需求選擇其他路徑）
    return paths[0] if paths else []

# ---------------------------
# 主循環：對每個 target 訓練模型、提取規則
predictions = {}
grid_ids = []
# tree_vectors = [] # 似乎未使用，可以考慮移除
geo_coords = []
#root_rules = {}    # 舊的根規則儲存，不再直接使用於分群
rule_paths = {}    # 舊的規則路徑儲存，新方法直接從樹結構提取特定規則

# 新增：記錄每個 target 的最佳 MAE、模型物件及最佳樹索引
target_mae = {}
target_mse = {}
target_models = {}
target_best_tree_index = {}
#%%
print(f"訓練單獨模型...")
for target in target_columns:
    train_data = lgb.Dataset(X_train_tree, label=y_train[target], categorical_feature=cat_features)
    test_data = lgb.Dataset(X_test_tree, label=y_test[target], reference=train_data, categorical_feature=cat_features)
    # metric同時計算 "l2" (MSE) 與 "l1" (MAE)
    params = {
        'objective': 'regression',
        'metric': ['l2', 'l1'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'seed': 42
    }
    evals_result = {}
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[test_data],
        valid_names=["valid_0"],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                   lgb.record_evaluation(evals_result),
                   lgb.log_evaluation(100)]
    )
    y_pred = lgb_model.predict(X_test_tree, num_iteration=lgb_model.best_iteration)
    predictions[target] = y_pred
    if "valid_0" in evals_result and "l2" in evals_result["valid_0"]:
        plt.figure(figsize=(8, 5))
        # 繪製 MSE 學習曲線
        plt.plot(evals_result['valid_0']['l2'], label="Validation MSE", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title(f"Learning Curve ({target})")
        plt.legend()
        learning_curve_path = os.path.join(result_dir, "learning_curve", f"{target.replace(',', '_').replace(' ', '')}.png")
        plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print(f"無法繪製 {target} 的學習曲線。")

    model_dict = lgb_model.dump_model()
    tree_info = model_dict["tree_info"]
    # 取每棵樹根節點的 split_gain，若不存在則設為 0
    split_gains = [tree_info[i]["tree_structure"].get("split_gain", 0) for i in range(len(tree_info))]
    best_tree_index = np.argmax(split_gains)
    # 使用 MAE 計算模型表現
    target_mae[target] = mean_absolute_error(y_test[target], y_pred)
    target_mse[target] = mean_squared_error(y_test[target], y_pred)
    target_models[target] = lgb_model
    target_best_tree_index[target] = best_tree_index

    plt.figure(figsize=(30, 18))
    lgb.plot_tree(lgb_model, tree_index=best_tree_index, show_info=['split_gain', 'data_count'])
    plt.title(f"Best Decision Tree for {target} (Highest split_gain)")
    tree_plot_path = os.path.join(result_dir, "tree", f"{target.replace(',', '_').replace(' ', '')}.png")
    plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_summary_path = os.path.join(result_dir, "shap_summary", f"{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    shap_bar_path = os.path.join(result_dir, "shap_bar", f"{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 提取最佳決策樹的根部規則（僅取根節點）
    best_tree = tree_info[best_tree_index]["tree_structure"]
    root_rule = (best_tree.get("split_feature"), best_tree.get("threshold"))
    #root_rules[target] = root_rule
    # 提取廣度優先規則路徑（整棵樹，廣度順序）
    # path = get_breadth_first_path(best_tree) # 舊的 get_breadth_first_path 不再直接用於分群
    # rule_paths[target] = path # 不再使用 rule_paths
    geo_coords.append(target) # geo_coords 應與 target_columns 一致
    grid_ids.append(target) # grid_ids 應與 target_columns 一致
    
    # TODO:
    # 統計每個決策規則:
    # 出現在根結點的規則，每出現一次+3分
    # 出現在第二層的規則，每出現一次+2分
    # 出現在第三層的規則，每出現一次+1分
    feature_names_from_model = list(X_train_tree.columns) # 這些是英文特徵名稱
    # cat_features 已經定義為英文名稱，例如 ['Holiday']
    #collect_rules_with_scores(best_tree, 1, 3, feature_names_from_model, cat_features, rule_statistics, reverse_mapping)
    #time.sleep(1)
#%%
print("\n開始匯出決策規則統計...")
excel_data_for_rules = []
for rule_desc, score in rule_statistics.items():
    excel_data_for_rules.append({
        "決策規則 (中文)": rule_desc,
        "總分數": score
    })

df_rule_stats = pd.DataFrame(excel_data_for_rules)
# 按分數降序排序
df_rule_stats = df_rule_stats.sort_values(by="總分數", ascending=False)

rule_stats_excel_path = os.path.join(result_dir, "decision_rule_statistics.xlsx")
try:
    df_rule_stats.to_excel(rule_stats_excel_path, index=False, engine='openpyxl')
    print(f"決策規則統計已儲存至: {rule_stats_excel_path}")
except ImportError:
    print("請安裝 'openpyxl' 套件以支援 Excel (.xlsx) 檔案匯出：pip install openpyxl")
    
    rule_stats_csv_path = os.path.join(result_dir, "decision_rule_statistics.csv")
    df_rule_stats.to_csv(rule_stats_csv_path, index=False, encoding='utf-8-sig')
    print(f"決策規則統計已儲存為 CSV 格式至: {rule_stats_csv_path}")

# 定義結果儲存目錄
result_dir = r"C:\\thesis\\code\\result_lgb"  # 請根據實際路徑調整
shared_result_dir = os.path.join(result_dir, "shared_model")

# 建立必要的子目錄
os.makedirs(os.path.join(shared_result_dir, 'learning_curve'), exist_ok=True)
os.makedirs(os.path.join(shared_result_dir, 'shap_summary'), exist_ok=True)
os.makedirs(os.path.join(shared_result_dir, 'shap_bar'), exist_ok=True)
os.makedirs(os.path.join(shared_result_dir, 'model'), exist_ok=True)
os.makedirs(os.path.join(shared_result_dir, 'tree'), exist_ok=True)

# 假設 X_train, X_test, y_train, y_test, target_columns 已定義
# 準備多輸出目標數據
y_combined_train = y_train.values  # 形狀: (train_samples, 495)
y_combined_test = y_test.values    # 形狀: (test_samples, 495)

def parse_coord_string(coord_str):
    coord_str = coord_str.strip("() ")
    lon_str, lat_str = coord_str.split(",")
    return float(lon_str), float(lat_str)

# 提取所有網格的經度和緯度
coords = [parse_coord_string(coord) for coord in target_columns]
lons = np.array([coord[0] for coord in coords])
lats = np.array([coord[1] for coord in coords])

# 重塑訓練數據
n_samples_train = X_train.shape[0]
n_targets = len(target_columns)
X_train_expanded = []
y_train_expanded = []

for i in range(n_targets):
    X_temp = X_train.copy()  # 不添加 lon 和 lat
    X_train_expanded.append(X_temp)
    y_train_expanded.append(y_train.iloc[:, i])

X_train_expanded = pd.concat(X_train_expanded, axis=0)
y_train_expanded = np.concatenate(y_train_expanded)

# 重塑測試數據
n_samples_test = X_test.shape[0]
X_test_expanded = []
y_test_expanded = []

for i in range(n_targets):
    X_temp = X_test.copy()  # 不添加 lon 和 lat
    X_test_expanded.append(X_temp)
    y_test_expanded.append(y_test.iloc[:, i])

X_test_expanded = pd.concat(X_test_expanded, axis=0)
y_test_expanded = np.concatenate(y_test_expanded)

# LightGBM 參數
shared_params = {
    'objective': 'regression',
    'metric': ['l2', 'l1'],
    'learning_rate': 0.1,
    'num_leaves': 63,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_iterations': 10000,
    'early_stopping_rounds': 10,
}

# 模型文件路徑
model_file_path = os.path.join(shared_result_dir, 'model', 'shared_model.txt')

# 檢查是否已有模型
# if os.path.exists(model_file_path):
#     print(f"載入已存在的模型: {model_file_path}")
#     shared_model = lgb.Booster(model_file=model_file_path)
# else:
# 獲取特徵名稱並轉換為英文
feature_names = X_train.columns.tolist()
english_feature_names = [feature_mapping.get(name, name) for name in feature_names]

# 創建 LightGBM 數據集並指定英文特徵名稱
train_data_shared = lgb.Dataset(X_train_expanded, label=y_train_expanded, 
                                feature_name=english_feature_names)
valid_data_shared = lgb.Dataset(X_test_expanded, label=y_test_expanded, 
                                reference=train_data_shared, 
                                feature_name=english_feature_names)
#%%
# 訓練共享模型
print("開始訓練共享模型以預測所有網格...")
evals_result = {}
shared_model = lgb.train(shared_params, train_data_shared, valid_sets=[valid_data_shared],
                            callbacks=[lgb.record_evaluation(evals_result)])

# 儲存模型
shared_model.save_model(model_file_path)
print(f"模型已儲存至: {model_file_path}")

# 預測
y_pred_expanded = shared_model.predict(X_test_expanded)

# 重塑預測結果為 (n_samples, n_targets)
y_pred_shared = y_pred_expanded.reshape(n_samples_test, n_targets)

# --- 學習曲線 ---
plt.figure(figsize=(10, 6))
plt.plot(evals_result['valid_0']['l2'], label='Validation MSE')
plt.xlabel('迭代次數')
plt.ylabel('誤差')
plt.title('共享模型的學習曲線')
plt.legend()
plt.savefig(os.path.join(shared_result_dir, 'learning_curve', 'shared_model.png'), dpi=300)
plt.close()

# --- SHAP 分析 ---
explainer = shap.TreeExplainer(shared_model)
shap_values = explainer.shap_values(X_test_expanded)

# 為每個網格生成 SHAP 圖
for i, target in enumerate(target_columns):
    safe_target = target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    idx = slice(i * n_samples_test, (i + 1) * n_samples_test)
    
    # SHAP 摘要圖（點圖）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[idx], X_test_expanded.iloc[idx], plot_type='dot', show=False)
    plt.title(f'共享模型 SHAP 摘要圖 ({target})')
    plt.savefig(os.path.join(shared_result_dir, 'shap_summary', f'shared_model_{safe_target}.png'), dpi=300)
    plt.close()
    
    # SHAP 特徵重要性圖（條形圖）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[idx], X_test_expanded.iloc[idx], plot_type='bar', show=False)
    plt.title(f'共享模型 SHAP 特徵重要性 ({target})')
    plt.savefig(os.path.join(shared_result_dir, 'shap_bar', f'shared_model_{safe_target}.png'), dpi=300)
    plt.close()

# 生成平均 SHAP 圖
mean_shap_values = np.mean(shap_values.reshape(n_targets, n_samples_test, -1), axis=0)
X_test_for_shap = X_test_expanded.iloc[:n_samples_test]

plt.figure(figsize=(10, 8))
shap.summary_plot(mean_shap_values, X_test_for_shap, plot_type='dot', show=False)
plt.title('共享模型平均 SHAP 摘要圖')
plt.savefig(os.path.join(shared_result_dir, 'shap_summary', 'shared_model_average.png'), dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(mean_shap_values, X_test_for_shap, plot_type='bar', show=False)
plt.title('共享模型平均 SHAP 特徵重要性')
plt.savefig(os.path.join(shared_result_dir, 'shap_bar', 'shared_model_average.png'), dpi=300)
plt.close()

# --- 評估指標 ---
shared_mse = {}
shared_mae = {}
for i, target in enumerate(target_columns):
    shared_mse[target] = mean_squared_error(y_combined_test[:, i], y_pred_shared[:, i])
    shared_mae[target] = mean_absolute_error(y_combined_test[:, i], y_pred_shared[:, i])

# 計算平均 MSE 和 MAE
average_mse = np.mean(list(shared_mse.values()))
average_mae = np.mean(list(shared_mae.values()))
print(f"共享模型平均 MSE: {average_mse:.2f}")
print(f"共享模型平均 MAE: {average_mae:.2f}")

# 儲存指標至 CSV
metrics_df = pd.DataFrame({
    'target': target_columns,
    'MSE': [shared_mse[target] for target in target_columns],
    'MAE': [shared_mae[target] for target in target_columns]
})
metrics_df = pd.concat([metrics_df, pd.DataFrame([{
    'target': '平均',
    'MSE': average_mse,
    'MAE': average_mae
}])], ignore_index=True)
metrics_file = os.path.join(shared_result_dir, 'shared_model_metrics.csv')
metrics_df.to_csv(metrics_file, index=False)
print(f"共享模型評估指標已儲存至: {metrics_file}")

# 繪製 split_gain 最高的決策樹
model_dict = shared_model.dump_model()
tree_info = model_dict["tree_info"]

# 提取每棵樹的 split_gain
split_gains = [tree["tree_structure"].get("split_gain", 0) for tree in tree_info]
best_tree_index = np.argmax(split_gains)

# 繪製最佳樹，顯示 split_gain 和資料筆數
plt.figure(figsize=(30, 18))
lgb.plot_tree(shared_model, tree_index=best_tree_index, 
              show_info=['split_gain', 'internal_count', 'leaf_count'])
plt.title(f"split_gain 最佳決策樹")
tree_plot_path = os.path.join(shared_result_dir, 'tree', 'best_tree.png')
plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
plt.close()
print(f"最佳決策樹圖已儲存至: {tree_plot_path}")

# --- 儲存決策樹規則 ---
def get_breadth_first_path(tree_structure):
    """以廣度優先順序遍歷樹，返回所有節點的 (split_feature, threshold) 規則"""
    path = []
    q = deque([tree_structure])
    while q:
        node = q.popleft()
        if "split_feature" in node:
            feature_idx = node.get("split_feature")
            threshold = node.get("threshold")
            rule = (feature_idx, threshold)
            path.append(rule)
            if "left_child" in node:
                q.append(node["left_child"])
            if "right_child" in node:
                q.append(node["right_child"])
    return path

# 提取最佳樹的規則路徑
best_tree = tree_info[best_tree_index]
best_tree_rules = get_breadth_first_path(best_tree["tree_structure"])

# 將規則儲存為文本文件
rules_file_path = os.path.join(shared_result_dir, 'tree', 'best_tree_rules.txt')
with open(rules_file_path, 'w', encoding='utf-8') as f:
    for rule in best_tree_rules:
        feature_idx, threshold = rule
        feature_name = X_train_expanded.columns[feature_idx]
        f.write(f"Feature: {feature_name}, Threshold: {threshold}\n")
print(f"最佳決策樹規則已儲存至: {rules_file_path}")
#%%
# ------------------------
# 新分群邏輯開始

# 輔助函數：從節點字典中提取規則 (特徵索引, 閾值)
def get_node_rule(node_dict):
    """從LGBM樹模型節點字典中提取分裂規則。"""
    if node_dict and "split_feature" in node_dict and node_dict.get("split_gain", 0) > 0: # 確保是有效分裂
        return (node_dict["split_feature"], node_dict["threshold"])
    return None

# 輔助函數：為單一目標提取 R1, R2_left, R3_right 規則
all_target_rules_info = {} # 儲存 {target: {'r1': rule, 'r2_left': rule, 'r3_right': rule}}
print("提取各目標點的 R1, R2_left, R3_right 規則...")
for target in target_columns:
    if target not in target_models: # 如果某目標沒有成功訓練模型
        all_target_rules_info[target] = {'r1': None, 'r2_left': None, 'r3_right': None}
        continue
    model_dump = target_models[target].dump_model()
    # 確保 tree_info 非空且 best_tree_index 有效
    if not model_dump["tree_info"] or target_best_tree_index[target] >= len(model_dump["tree_info"]):
        # print(f"警告: 目標 {target} 的 tree_info 為空或 best_tree_index 無效。")
        all_target_rules_info[target] = {'r1': None, 'r2_left': None, 'r3_right': None}
        continue
    
    best_tree_structure = model_dump["tree_info"][target_best_tree_index[target]]["tree_structure"]

    r1 = get_node_rule(best_tree_structure)
    r2_left = get_node_rule(best_tree_structure.get("left_child"))
    r3_right = get_node_rule(best_tree_structure.get("right_child"))
    all_target_rules_info[target] = {'r1': r1, 'r2_left': r2_left, 'r3_right': r3_right}

# 輔助函數：合併相似規則並計數
def merge_individual_rules_and_count(target_to_rule_dict, cat_features_eng, threshold_dist, feature_names_eng_list):
    """
    合併相似規則並計數。
    target_to_rule_dict: {target_id: (feat_idx, thresh)}
    cat_features_eng: 英文類別特徵名稱列表
    threshold_dist: 數值特徵閾值的合併距離
    feature_names_eng_list: X_train_tree.columns (英文特徵名稱列表)
    返回: merged_rule_to_count (合併後規則 -> 計數), target_to_merged_rule (目標 -> 合併後規則)
    """
    merged_rule_representatives = []  # 儲存每個合併群組的代表規則
    merged_rule_to_count = {}
    target_to_merged_rule = {}

    for target_id, rule in target_to_rule_dict.items():
        if rule is None:
            target_to_merged_rule[target_id] = None # 保留 None 規則
            merged_rule_to_count[None] = merged_rule_to_count.get(None, 0) + 1
            if None not in merged_rule_representatives and None not in [r for r in merged_rule_representatives if r is None]: # 確保 None 只添加一次
                 if not any(r is None for r in merged_rule_representatives): # 修正檢查方式
                    merged_rule_representatives.append(None)
            continue

        feat_idx, thresh = rule
        # 檢查 feat_idx 是否在範圍內
        if feat_idx >= len(feature_names_eng_list):
            # print(f"警告: 特徵索引 {feat_idx} 超出範圍。目標 {target_id} 的此規則將被視為 None。")
            target_to_merged_rule[target_id] = None
            merged_rule_to_count[None] = merged_rule_to_count.get(None, 0) + 1
            if not any(r is None for r in merged_rule_representatives):
                 merged_rule_representatives.append(None)
            continue

        current_feature_name_eng = feature_names_eng_list[feat_idx]
        
        matched_representative = None
        for rep_rule in merged_rule_representatives:
            if rep_rule is None:
                continue
            rep_feat_idx, rep_thresh = rep_rule
            rep_feature_name_eng = feature_names_eng_list[rep_feat_idx]

            if feat_idx == rep_feat_idx: # 特徵索引必須相同
                is_categorical = current_feature_name_eng in cat_features_eng
                if is_categorical:
                    if str(thresh) == str(rep_thresh): # LightGBM 對類別特徵的閾值可能是集合字串
                        matched_representative = rep_rule
                        break
                else: # 數值特徵
                    try:
                        if abs(float(thresh) - float(rep_thresh)) <= threshold_dist:
                            matched_representative = rep_rule
                            break
                    except (TypeError, ValueError): # 轉換失敗，視為不匹配
                        pass
        
        if matched_representative is not None:
            target_to_merged_rule[target_id] = matched_representative
            merged_rule_to_count[matched_representative] = merged_rule_to_count.get(matched_representative, 0) + 1
        else:
            target_to_merged_rule[target_id] = rule
            merged_rule_to_count[rule] = merged_rule_to_count.get(rule, 0) + 1
            merged_rule_representatives.append(rule)
            
    return merged_rule_to_count, target_to_merged_rule

# 輔助函數：尋找最常見的規則
def find_most_frequent_rule(merged_rule_to_count):
    """從合併後的規則計數字典中找出最常見的規則。"""
    if not merged_rule_to_count:
        return None
    # 排除 None 規則的計數，除非它是唯一的選擇
    valid_rules = {r: c for r, c in merged_rule_to_count.items() if r is not None}
    if not valid_rules: # 如果只有 None 規則或字典為空
        return None # 或者 max(merged_rule_to_count, key=merged_rule_to_count.get) 如果允許 None 為最常見
    return max(valid_rules, key=valid_rules.get)


# 分層分群主邏輯
initial_targets = list(target_columns)
# 每個元素是 (targets_list, defining_rules_dict)
# defining_rules_dict 結構: {'r1': rule_tuple, 'r2_left': rule_tuple, 'r3_right': rule_tuple}
processing_groups = [(initial_targets, {})] 
final_eight_groups_details = []

rule_keys_for_splitting = ['r1', 'r2_left', 'r3_right']
feature_names_english = list(X_train_tree.columns) # 用於 merge_individual_rules_and_count
#%%
print("開始分層分群...")
for i, rule_key in enumerate(rule_keys_for_splitting):
    print(f"  處理分群層級: {rule_key} (第 {i+1} 層)")
    next_level_processing_groups = []
    for current_group_targets, path_rules_so_far in processing_groups:
        if not current_group_targets: # 如果當前群組已空，直接帶入下一層
            # 確保即使群組為空，也為其分配一個定義規則（例如 None）以保持結構
            next_level_processing_groups.append(([], {**path_rules_so_far, rule_key: None}))
            next_level_processing_groups.append(([], {**path_rules_so_far, rule_key: None}))
            continue

        # 1. 提取當前群組中所有目標在此分裂層級的規則
        rules_at_this_level_for_group = {
            t: all_target_rules_info[t][rule_key] for t in current_group_targets
        }
        
        # 2. 合併這些規則並計數
        # 注意：cat_features 是全局定義的英文列表
        merged_counts, target_to_merged_rule_map = merge_individual_rules_and_count(
            rules_at_this_level_for_group, cat_features, 1.5, feature_names_english
        )

        # 3. 找出最常見的合併後規則 (Type A)
        defining_rule_A = find_most_frequent_rule(merged_counts)

        # 4. 分裂目標點
        group_A_targets = []
        group_B_targets_candidates = []

        for t in current_group_targets:
            merged_rule_for_t = target_to_merged_rule_map.get(t)
            if merged_rule_for_t == defining_rule_A:
                group_A_targets.append(t)
            else:
                group_B_targets_candidates.append(t)
        
        # 5. 為 Group B (候選者) 找出其代表規則 (Type B)
        defining_rule_B = None
        if group_B_targets_candidates:
            rules_for_group_B_candidates = {
                t: all_target_rules_info[t][rule_key] for t in group_B_targets_candidates
            }
            merged_counts_B, _ = merge_individual_rules_and_count( # target_to_merged_map_B 不直接使用
                rules_for_group_B_candidates, cat_features, 1.5, feature_names_english
            )
            defining_rule_B = find_most_frequent_rule(merged_counts_B)
            # 如果 defining_rule_B 與 defining_rule_A 相同（可能發生在 B 組很小或規則單一時），
            # 且 defining_rule_A 不是 None，則嘗試選擇 B 組中次常見的，或保持 B 組為空，所有點歸入 A。
            # 簡化處理：如果 B 組的 "最常見" 與 A 組的 "最常見" 相同，則 B 組的定義規則就是這個。
            # 如果 B 組沒有有效規則，defining_rule_B 會是 None。
        
        # 如果 defining_rule_A 是 None，所有非 None 規則的目標點都應進入 B 組
        if defining_rule_A is None and group_B_targets_candidates:
             # 此時 A 組是那些規則為 None 的，B 組是那些規則不為 None 的
             # B 組的定義規則應基於 group_B_targets_candidates 中的最常見規則
             pass # defining_rule_B 已經計算過了

        # 添加到下一層處理列表
        next_level_processing_groups.append((group_A_targets, {**path_rules_so_far, rule_key: defining_rule_A}))
        next_level_processing_groups.append((group_B_targets_candidates, {**path_rules_so_far, rule_key: defining_rule_B}))

    processing_groups = next_level_processing_groups

final_eight_groups_details = processing_groups
print(f"分群完成，共形成 {len(final_eight_groups_details)} 個最終群組。")
#%%
# 輔助函數：將規則元組轉換為人類可讀的字串
def format_rule_to_string(rule_tuple, feature_names_eng_list, reverse_mapping_dict, cat_features_eng_list):
    if rule_tuple is None:
        return "N/A"
    try:
        feat_idx, threshold_val = rule_tuple
        if feat_idx >= len(feature_names_eng_list):
            return f"錯誤索引({feat_idx})"
        
        feature_name_eng = feature_names_eng_list[feat_idx]
        feature_name_chi = reverse_mapping_dict.get(feature_name_eng, feature_name_eng)
        
        operator_str = ""
        processed_threshold_str = ""

        if feature_name_eng in cat_features_eng_list:
            operator_str = "=="
            # LightGBM dump_model 對於類別特徵的閾值可能是 'valueA||valueB' 格式
            processed_threshold_str = str(threshold_val)
        else: # 數值特徵
            operator_str = "<="
            try:
                processed_threshold_str = f"{float(threshold_val):.1f}"
            except (ValueError, TypeError):
                processed_threshold_str = str(threshold_val) # 若轉換失敗，保留原始
        return f"{feature_name_chi} {operator_str} {processed_threshold_str}"
    except Exception as e:
        # print(f"格式化規則時出錯: {rule_tuple}, 錯誤: {e}")
        return "格式化錯誤"

# 整理並匯出分群結果
output_data = []
for i, (targets_in_group, group_rules_dict) in enumerate(final_eight_groups_details):
    # 確保所有規則鍵都存在
    r1_str = format_rule_to_string(group_rules_dict.get('r1'), feature_names_english, reverse_mapping, cat_features)
    r2_str = format_rule_to_string(group_rules_dict.get('r2_left'), feature_names_english, reverse_mapping, cat_features)
    r3_str = format_rule_to_string(group_rules_dict.get('r3_right'), feature_names_english, reverse_mapping, cat_features)
    
    output_data.append({
        "組別ID": i + 1,
        "規則1 (R1)": r1_str,
        "規則2 (R2_left)": r2_str,
        "規則3 (R3_right)": r3_str,
        "組內座標數量": len(targets_in_group),
        "組內所有座標": ", ".join(sorted(targets_in_group)) if targets_in_group else ""
    })

output_df = pd.DataFrame(output_data)
output_csv_path = os.path.join(result_dir, "hierarchical_grouping_results.csv")
output_excel_path = os.path.join(result_dir, "hierarchical_grouping_results.xlsx")

output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"階層式分群結果已儲存至 CSV: {output_csv_path}")
try:
    output_df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"階層式分群結果已儲存至 Excel: {output_excel_path}")
except ImportError:
    print("警告: 未安裝 'openpyxl'。Excel 檔案未儲存。請執行 'pip install openpyxl'")


# 地理分佈視覺化
# 建立 target -> 組別ID 的映射
target_to_group_id_map = {}
for i, (targets_in_group, _) in enumerate(final_eight_groups_details):
    for target_id_str in targets_in_group:
        target_to_group_id_map[target_id_str] = i # 組別 ID 從 0 開始，方便色彩映射

# 準備繪圖數據
plot_lons = []
plot_lats = []
plot_group_labels = []

# target_columns 已經是排序且唯一的列表
for target_coord_str in target_columns:
    lon, lat = parse_coord_string(target_coord_str) # 假設 parse_coord_string 已定義
    plot_lons.append(lon)
    plot_lats.append(lat)
    # 如果某個 target_column 中的點未被分到任何組（理論上不應發生，除非初始 target_columns 與模型訓練的不一致）
    # 則給定一個特殊標籤，例如 -1
    plot_group_labels.append(target_to_group_id_map.get(target_coord_str, -1))


# 定義顏色映射，嘗試使相似規則的群組顏色相近
# 8個群組，可以基於 R1 的兩種主要類型來選擇基礎色調
# 例如，R1_typeA (通常是第一個分裂出的) 用藍色系，R1_typeB 用紅色系
# 然後 R2_left 的類型調整深淺，R3_right 再調整
# 這裡使用一個定性的 colormap，它能提供8種區分明顯的顏色
# cmap = plt.cm.get_cmap('viridis', 8) # viridis 可能不夠區分
# cmap = plt.cm.get_cmap('tab10', 8) # tab10 有10種顏色，取前8種
# 或者手動指定顏色以更好地控制相似性
# 假設 group_id 0-3 源於 R1 的第一種主要分裂，4-7 源於第二種
colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',  # R1_typeA (藍/橙系)
    '#2ca02c', '#98df8a', '#d62728', '#ff9896'   # R1_typeB (綠/紅系)
]
# 如果 final_eight_groups_details 的順序確實反映了分裂層次，這個顏色列表可以直接用
# 否則，需要根據 group_rules_dict 中的 R1, R2, R3 類型來動態決定顏色

plt.figure(figsize=(12, 10))
scatter = plt.scatter(plot_lons, plot_lats, c=plot_group_labels, cmap=plt.cm.colors.ListedColormap(colors[:len(set(plot_group_labels))]), s=50, alpha=0.8, vmin=-0.5, vmax=7.5)
plt.ticklabel_format(useOffset=False, style='plain', axis='both')
plt.xlabel("經度 (Longitude)")
plt.ylabel("緯度 (Latitude)")
plt.title("階層式決策規則分群之地理分佈")

# 創建圖例
handles, labels = scatter.legend_elements(prop="colors", num=None) # num=None 或指定組數
legend_labels = [f"組別 {i+1}" for i in sorted(list(set(plot_group_labels)))] # 確保標籤與實際出現的組對應
if -1 in set(plot_group_labels): # 如果有未分組的點
    legend_labels = [lbl if not lbl.endswith("-0") else "未分組" for lbl in legend_labels]


# 確保圖例標籤數量與handles數量一致
if len(handles) == len(legend_labels):
    plt.legend(handles, legend_labels, title="群組")
else:
    # print(f"警告: 圖例handles數量 ({len(handles)}) 與標籤數量 ({len(legend_labels)}) 不符。可能不會顯示圖例。")
    # 備用圖例，如果 scatter.legend_elements() 的 num 控制不理想
    unique_labels = sorted(list(set(plot_group_labels)))
    if len(unique_labels) <= len(colors): # 確保顏色足夠
        custom_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in unique_labels if i != -1]
        custom_labels = [f"組別 {i+1}" for i in unique_labels if i != -1]
        if -1 in unique_labels:
            custom_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10)) # 給未分組的一個顏色
            custom_labels.append("未分組")
        plt.legend(custom_handles, custom_labels, title="群組")


grouping_plot_path = os.path.join(result_dir, "geo_hierarchical_grouping.png")
plt.savefig(grouping_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"階層式分群地理分佈圖已儲存至: {grouping_plot_path}")


# 移除舊的 group_tree 繪圖部分，因為現在是8個固定群組，其定義規則已在CSV中
# 如果需要為每個群組的代表（例如，MAE最低的點）繪製其原始樹，可以另行添加
# 但題目要求是將8個群組繪製在地圖上，這已完成。

# 清理不再使用的舊分群相關變數和函數的定義（如果它們僅用於舊分群）
# 例如 assign_group_by_feature_prefix, merge_similar_groups, assign_group_by_decision_rules
# 以及它們的相關呼叫。由於這是替換，這裡不顯式刪除，假設它們在原始碼中被新邏輯取代。

print("所有處理完成。")
# %%

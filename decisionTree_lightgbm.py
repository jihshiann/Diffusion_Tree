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

def assign_group_by_feature_prefix(rule_paths, threshold):
    """
    根據每個 target 的規則路徑（只取 split_feature）進行分組，
    若某前綴下 target 數超過 threshold，則嘗試用更長前綴細分。
    """
    groups_temp = {}
    for target, path in rule_paths.items():
        feature_path = tuple(rule[0] for rule in path)
        for k in range(1, len(feature_path)+1):
            prefix = feature_path[:k]
            groups_temp.setdefault(prefix, []).append(target)
    final_groups = {}
    for target, path in rule_paths.items():
        feature_path = tuple(rule[0] for rule in path)
        assigned_prefix = feature_path  # 預設使用完整前綴
        for k in range(1, len(feature_path)+1):
            prefix = feature_path[:k]
            if len(groups_temp[prefix]) <= threshold:
                assigned_prefix = prefix
                break
        final_groups[target] = assigned_prefix
    return final_groups

# 合併相似的規則前綴
def merge_similar_groups(final_groups, threshold_distance=1.5):
    merged_groups = {}
    group_counts = {}
    for target, prefix in final_groups.items():
        merged_prefix = prefix
        for other_prefix in group_counts:
            if len(prefix) == len(other_prefix) and all(
                p[0] == o[0] and (p[0] in cat_features or abs(float(p[1]) - float(o[1])) <= threshold_distance)
                for p, o in zip(prefix, other_prefix)
            ):
                merged_prefix = other_prefix
                break
        merged_groups[target] = merged_prefix
        group_counts[merged_prefix] = group_counts.get(merged_prefix, 0) + 1
    return merged_groups

def assign_group_by_decision_rules(rule_paths, threshold):
    """
    根據完整的決策規則路徑（包含特徵和閾值）進行分群，
    若某路徑下的目標數超過 threshold，則使用更長的路徑細分。
    """
    groups_temp = {}
    for target, path in rule_paths.items():
        rule_tuple = tuple(path)  # 將路徑轉為 tuple 以作為 key
        for k in range(1, len(rule_tuple) + 1):
            prefix = rule_tuple[:k]
            groups_temp.setdefault(prefix, []).append(target)
    
    final_groups = {}
    for target, path in rule_paths.items():
        rule_tuple = tuple(path)
        assigned_prefix = rule_tuple  # 預設使用完整路徑
        for k in range(1, len(rule_tuple) + 1):
            prefix = rule_tuple[:k]
            if len(groups_temp[prefix]) <= threshold:
                assigned_prefix = prefix
                break
        final_groups[target] = assigned_prefix
    return final_groups

print(len(target_columns))

def collect_rules_with_scores(node, current_depth, max_depth, feature_names_model, model_cat_features, stats_dict, reverse_mapping_dict):
    """
    遞迴收集決策樹前 max_depth 層的規則，並根據深度給予分數。
    根節點(深度1)規則+3分，第二層(深度2)+2分，第三層(深度3)+1分。
    """
    if current_depth > max_depth or "split_feature" not in node:
        return

    feature_idx = node["split_feature"]
    raw_threshold = node["threshold"]
    
    # 確保 feature_idx 在邊界內
    if feature_idx >= len(feature_names_model):
        # print(f"警告：特徵索引 {feature_idx} 超出特徵名稱列表的範圍 (長度 {len(feature_names_model)})")
        return

    feature_name_eng = feature_names_model[feature_idx]
    
    score_to_add = 0
    if current_depth == 1:
        score_to_add = 3
    elif current_depth == 2:
        score_to_add = 2
    elif current_depth == 3:
        score_to_add = 1

    operator_str = ""
    processed_threshold_val = raw_threshold

    if feature_name_eng in model_cat_features:
        operator_str = "=="
        # 對於類別特徵，閾值通常是其本身的值。
        # LightGBM 的 dump_model 可能會將集合表示為 "valueA||valueB"
        processed_threshold_val = str(raw_threshold) # 確保是字串以保持一致性
    else:
        operator_str = "<="
        try:
            processed_threshold_val = round(float(raw_threshold), 1)
        except (ValueError, TypeError):
            # 如果轉換失敗，保留原始值（理論上不應發生於數值型分裂）
            pass 
            
    # 使用中文特徵名稱建立規則描述
    feature_name_chi = reverse_mapping_dict.get(feature_name_eng, feature_name_eng)
    rule_description = f"{feature_name_chi} {operator_str} {processed_threshold_val}"
    
    stats_dict[rule_description] = stats_dict.get(rule_description, 0) + score_to_add

    if "left_child" in node:
        collect_rules_with_scores(node["left_child"], current_depth + 1, max_depth, feature_names_model, model_cat_features, stats_dict, reverse_mapping_dict)
    
    if "right_child" in node:
        collect_rules_with_scores(node["right_child"], current_depth + 1, max_depth, feature_names_model, model_cat_features, stats_dict, reverse_mapping_dict)

    # ---------------------------
# 主循環：對每個 target 訓練模型、提取規則
predictions = {}
grid_ids = []
tree_vectors = []
geo_coords = []
root_rules = {}    # 儲存每個 target 的根部規則 (僅根節點)
rule_paths = {}    # 儲存每個 target 的廣度優先規則路徑

# 新增：記錄每個 target 的最佳 MAE、模型物件及最佳樹索引
target_mae = {}
target_mse = {}
target_models = {}
target_best_tree_index = {}

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
    root_rules[target] = root_rule
    # 提取廣度優先規則路徑（整棵樹，廣度順序）
    path = get_breadth_first_path(best_tree)
    #path = get_decision_path(best_tree, max_depth=3)
    rule_paths[target] = path
    geo_coords.append(target)
    grid_ids.append(target)
    
    # TODO:
    # 統計每個決策規則:
    # 出現在根結點的規則，每出現一次+3分
    # 出現在第二層的規則，每出現一次+2分
    # 出現在第三層的規則，每出現一次+1分
    feature_names_from_model = list(X_train_tree.columns) # 這些是英文特徵名稱
    # cat_features 已經定義為英文名稱，例如 ['Holiday']
    collect_rules_with_scores(best_tree, 1, 3, feature_names_from_model, cat_features, rule_statistics, reverse_mapping)
    #time.sleep(1)

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

# ------------------------
# 分群規則：
total_targets = len(target_columns)
threshold = 100 

#final_groups = assign_group_by_feature_prefix(rule_paths, threshold)
final_groups = assign_group_by_decision_rules(rule_paths, threshold)
final_groups = merge_similar_groups(final_groups, threshold_distance=2)

# 將每個群中 MAE 最低的座標選為群代表
group_to_targets = {}
for target, group_prefix in final_groups.items():
    group_to_targets.setdefault(group_prefix, []).append(target)

group_representative = {}
# 這裡使用 target_mae 作為選擇群代表的標準
for group_prefix, targets in group_to_targets.items():
    best_target = min(targets, key=lambda t: target_mae[t])
    group_representative[group_prefix] = best_target

# 將群代表的模型視覺化，同時標示出該群的規則
for group_prefix, rep_target in group_representative.items():
    model = target_models[rep_target]
    best_tree_index = target_best_tree_index[rep_target]
    
    # 將英文規則轉為中文，並保留閾值與運算符
    rule_features = []
    for rule in group_prefix:
        feature_idx, threshold = rule
        feature_name = list(X_train_tree.columns)[feature_idx] if feature_idx < len(X_train_tree.columns) else str(feature_idx)
        feature_ch = reverse_mapping.get(feature_name, feature_name)
        condition = f"== {threshold}" if feature_name in cat_features else f"<= {float(threshold):.1f}"
        rule_features.append(f"{feature_ch} {condition}")
    rule_text = f"群代表座標: {rep_target}\n群規則: " + "; ".join(rule_features)
    
    # 繪製決策樹圖，並在圖上標示解釋文字
    plt.figure(figsize=(30, 18))
    lgb.plot_tree(shared_model, tree_index=0, show_info=['split_gain', 'data_count'])
    plt.suptitle(rule_text, fontsize=10)
    safe_target = rep_target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    rep_tree_plot_path = os.path.join(result_dir, "group_tree", f"group_representative_tree_{safe_target}.png")
    plt.savefig(rep_tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"群代表 {rep_target} 的決策樹圖已存至: {rep_tree_plot_path}")

# 建立 target -> 分組標籤對照表：將唯一前綴映射到數值標籤
unique_prefixes = {v for v in final_groups.values()}
prefix_to_label = {prefix: idx for idx, prefix in enumerate(unique_prefixes)}
group_labels = {target: prefix_to_label[final_groups[target]] for target in final_groups}

# 輸出分群結果到 CSV：顯示完整的中文規則（特徵名稱 + 閾值 + 運算符）
group_rows = []
for prefix, label in prefix_to_label.items():
    # 將英文規則轉為中文
    rules_str = []
    for rule in prefix:
        feature_idx, threshold = rule
        feature_name = list(X_train_tree.columns)[feature_idx] if feature_idx < len(X_train_tree.columns) else str(feature_idx)
        feature_ch = reverse_mapping.get(feature_name, feature_name)
        condition = f"== {threshold}" if feature_name in cat_features else f"<= {float(threshold):.1f}"
        rules_str.append(f"{feature_ch} {condition}")
    prefix_str = "; ".join(rules_str)
    targets_in_prefix = [t for t, p in final_groups.items() if p == prefix]
    count = len(targets_in_prefix)
    rep_target = group_representative[prefix]
    
    # 計算 MAE 與 MSE 的群體指標
    rep_mae = target_mae[rep_target]
    rep_mse = target_mse[rep_target]
    group_mae = np.mean([target_mae[t] for t in targets_in_prefix])
    group_mse = np.mean([target_mse[t] for t in targets_in_prefix])
    overall_mae = np.mean(list(target_mae.values()))
    overall_mse = np.mean(list(target_mse.values()))
    
    group_rows.append({
        "規則": prefix_str,
        "座標數": count,
        "分組標籤": label,
        "群代表座標": rep_target,
        "代表座標MAE": rep_mae,
        "群平均MAE": group_mae,
        "總平均MAE": overall_mae,
        "代表座標MSE": rep_mse,
        "群平均MSE": group_mse,
        "總平均MSE": overall_mse,
        "目標": ", ".join(targets_in_prefix)
    })

group_df = pd.DataFrame(group_rows)
excel_path = os.path.join(result_dir, "grouping_results.csv")
group_df.to_csv(excel_path, index=False, encoding='utf-8-sig')
print("分群結果已儲存至:", excel_path)

# 視覺化：將 geo_coords (存放 target 字串，格式 "(lon, lat)") 轉為數值型 tuple
parsed_coords = [tuple(map(float, coord.strip("() ").split(","))) for coord in target_columns]
group_label_list = [group_labels[t] for t in grid_ids]
all_lons = [coord[0] for coord in parsed_coords]
all_lats = [coord[1] for coord in parsed_coords]

plt.figure(figsize=(10, 8))
plt.scatter(all_lons, all_lats, c=group_label_list, cmap='viridis', s=50, alpha=0.7)
plt.ticklabel_format(useOffset=False, style='plain', axis='both')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("基於決策樹規則分群的地理分布")
plt.colorbar(label="Group Label")
grouping_path = os.path.join(result_dir, "geo_grouping_by_decision_rules.png")
plt.savefig(grouping_path, dpi=300, bbox_inches="tight")
plt.close()
print("基於決策樹規則的地理分群圖已儲存至:", grouping_path)

# 建立儲存群代表決策樹的子目錄
group_tree_dir = os.path.join(result_dir, "group_tree")
os.makedirs(group_tree_dir, exist_ok=True)

# 對每個群代表進行繪圖與標示群規則
for prefix, rep_target in group_representative.items():
    # 將英文規則轉為中文
    rule_features = []
    for rule in prefix:
        feature_idx, threshold = rule
        feature_name = list(X_train_tree.columns)[feature_idx] if feature_idx < len(X_train_tree.columns) else str(feature_idx)
        feature_ch = reverse_mapping.get(feature_name, feature_name)
        condition = f"== {threshold}" if feature_name in cat_features else f"<= {float(threshold):.1f}"
        rule_features.append(f"{feature_ch} {condition}")
    rule_str = "; ".join(rule_features)
    
    # 取得該群代表的決策樹模型
    model = target_models[rep_target]
    plt.figure(figsize=(30, 18))
    # 繪製決策樹
    lgb.plot_tree(
        model, 
        tree_index=target_best_tree_index[rep_target], 
        show_info=['split_gain', 'data_count'],
        graph_attr={
            'ranksep': '0.75',  # 層與層之間的距離（預設約 0.75）
            'nodesep': '0.25'   # 同層節點之間的距離（預設約 0.25）
        })
    # 在圖上標題處加入群代表與群規則說明
    plt.suptitle(f"群代表: {rep_target}\n群規則: {rule_str}", fontsize=5)
    safe_target = rep_target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    group_tree_path = os.path.join(group_tree_dir, f"group_representative_tree_{safe_target}.png")
    plt.savefig(group_tree_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"群代表 {rep_target} 的標註規則決策樹圖已存至: {group_tree_path}")
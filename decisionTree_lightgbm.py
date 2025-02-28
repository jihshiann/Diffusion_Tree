import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import train_test_split

# ---------------------------
# 設定與數據讀取
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = r"C:\thesis\code\Taipei_5x5\all_merged_5X5.csv"
df = pd.read_csv(data_path)
result_dir = r"C:\thesis\code\result"
os.makedirs(result_dir, exist_ok=True)

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
    '年': 'Year',
    '月': 'Month',
    '日': 'Day',
    '時': 'Hour'
}

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

cat_features = ['Holiday', 'Month']

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
    """以廣度優先順序遍歷樹，返回所有節點的 (split_feature, threshold) 規則"""
    path = []
    q = deque([tree_structure])
    while q:
        node = q.popleft()
        if "split_feature" in node:
            rule = (node.get("split_feature"), node.get("threshold"))
            path.append(rule)
            if "left_child" in node:
                q.append(node["left_child"])
            if "right_child" in node:
                q.append(node["right_child"])
    return path

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
        assigned_prefix = feature_path  # 預設使用完整路徑
        for k in range(1, len(feature_path)+1):
            prefix = feature_path[:k]
            if len(groups_temp[prefix]) <= threshold:
                assigned_prefix = prefix
                break
        final_groups[target] = assigned_prefix
    return final_groups

# ---------------------------
# 主循環：對每個 target 訓練模型、提取規則

predictions = {}
grid_ids = []
tree_vectors = []
geo_coords = []
root_rules = {}    # 儲存每個 target 的根部規則 (僅根節點)
rule_paths = {}    # 儲存每個 target 的廣度優先規則路徑

for target in target_columns:
    print(f"訓練與預測 {target} 的決策樹...")
    train_data = lgb.Dataset(X_train_tree, label=y_train[target], categorical_feature=cat_features)
    test_data = lgb.Dataset(X_test_tree, label=y_test[target], reference=train_data, categorical_feature=cat_features)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'seed': 42
    }
    evals_result = {}
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        valid_names=["valid_0"],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                   lgb.record_evaluation(evals_result),
                   lgb.log_evaluation(10)]
    )
    y_pred = lgb_model.predict(X_test_tree, num_iteration=lgb_model.best_iteration)
    predictions[target] = y_pred
    print(f"LightGBM 總樹數: {lgb_model.num_trees()}")
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
        print(f"無法繪製 {target} 的學習曲線。")
    model_dict = lgb_model.dump_model()
    tree_info = model_dict["tree_info"]
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
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_summary_path = os.path.join(result_dir, f"shap_summary_{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP 特徵重要性圖已儲存至: {shap_summary_path}")
    shap_bar_path = os.path.join(result_dir, f"shap_bar_{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP 特徵影響條形圖已儲存至: {shap_bar_path}")
    # 提取最佳決策樹的根部規則（僅取根節點）
    best_tree = tree_info[best_tree_index]["tree_structure"]
    root_rule = (best_tree.get("split_feature"), best_tree.get("threshold"))
    root_rules[target] = root_rule
    # 提取廣度優先規則路徑（整棵樹，廣度順序）
    path = get_breadth_first_path(best_tree)
    rule_paths[target] = path
    # 直接取該 target 的代表經緯度（target 本身為座標字串）
    geo_coords.append(target)
    grid_ids.append(target)

# ------------------------
# 分群：拓展規則
# 規則：
# 1) 初步以根部規則 (split_feature) 分組
# 2) 如果某組 target 數量超過總數的 1/10，
#    則對該組 target 進一步依據廣度優先規則路徑的前綴細分分組（只使用 split_feature，不考慮 threshold）
total_targets = len(target_columns)
threshold = total_targets / 10.0

def assign_group_by_feature_prefix(rule_paths, threshold):
    groups_temp = {}
    for target, path in rule_paths.items():
        feature_path = tuple(rule[0] for rule in path)  # 僅取 split_feature 部分
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

final_groups = assign_group_by_feature_prefix(rule_paths, threshold)

# 建立 target -> 分組標籤對照表：將唯一前綴映射到數值標籤
unique_prefixes = {v for v in final_groups.values()}
prefix_to_label = {prefix: idx for idx, prefix in enumerate(unique_prefixes)}
group_labels = {target: prefix_to_label[final_groups[target]] for target in final_groups}

# 輸出分群結果到 CSV：僅顯示分組時使用的特徵名稱 (中文)
reverse_mapping = {v: k for k, v in feature_mapping.items()}
group_rows = []
for prefix, label in prefix_to_label.items():
    rules_str = []
    for feature_idx in prefix:
        if feature_idx < len(X_train_tree.columns):
            feature_eng = list(X_train_tree.columns)[feature_idx]
        else:
            feature_eng = str(feature_idx)
        feature_ch = reverse_mapping.get(feature_eng, feature_eng)
        rules_str.append(f"{feature_ch}")
    prefix_str = "; ".join(rules_str)
    targets_in_prefix = [t for t, p in final_groups.items() if p == prefix]
    count = len(targets_in_prefix)  # 新增座標數
    group_rows.append({
        "規則": prefix_str,
        "座標數": count,
        "分組標籤": label,
        "目標": ", ".join(targets_in_prefix)
    })

group_df = pd.DataFrame(group_rows)
excel_path = os.path.join(result_dir, "grouping_results.csv")
group_df.to_csv(excel_path, index=False, encoding='utf-8-sig')
print("分群結果已儲存至:", excel_path)

# 視覺化：將 geo_coords (存放 target 字串，格式 "(lon, lat)") 轉為數值型 tuple
parsed_coords = [tuple(map(float, coord.strip("() ").split(","))) for coord in target_columns]
# 建立分組標籤列表，依據 grid_ids (grid_ids 存放 target)
group_label_list = [group_labels[t] for t in grid_ids]
all_lons = [coord[0] for coord in parsed_coords]
all_lats = [coord[1] for coord in parsed_coords]

plt.figure(figsize=(10, 8))
plt.scatter(all_lons, all_lats, c=group_label_list, cmap='viridis', s=50, alpha=0.7)
plt.ticklabel_format(useOffset=False, style='plain', axis='both')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("基於決策樹根部規則(廣度分層)分群的地理分布")
plt.colorbar(label="Group Label")
grouping_path = os.path.join(result_dir, "geo_grouping_by_root_feature_refined.png")
plt.savefig(grouping_path, dpi=300, bbox_inches="tight")
plt.close()
print("基於決策樹根部規則(廣度分層)的地理分群圖已儲存至:", grouping_path)

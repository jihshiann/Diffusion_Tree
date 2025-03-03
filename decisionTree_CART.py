import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import shap

# ---------------------------
# 設定與資料讀取
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = r"C:\thesis\code\Taipei_5x5\all_merged_5X5.csv"
df = pd.read_csv(data_path)
result_dir = r"C:\thesis\code\result_cart"
os.makedirs(result_dir, exist_ok=True)

# 處理角度變數：對「最大陣風風向」和「風向」分別計算正弦與餘弦值
for col in ['最大陣風風向', '風向']:
    df[f'sin_{col}'] = np.sin(np.deg2rad(df[col]))
    df[f'cos_{col}'] = np.cos(np.deg2rad(df[col]))

# 處理類別特徵：將 'hoilday' 與 '月' 轉換為類別型並以 .cat.codes 處理
for col in ['hoilday', '月']:
    df[col] = df[col].astype('category').cat.codes

# 建立特徵名稱翻譯對應表
feature_mapping = {
    '測站氣壓': 'Station_Pressure',
    '海平面氣壓': 'Sea_Level_Pressure',
    '氣溫': 'Temperature',
    '露點溫度': 'Dew_Point',
    '相對溼度': 'Relative_Humidity',
    '風速': 'Wind_Speed',
    '風向': 'Wind_Direction',
    'sin_最大陣風風向': 'sin_Max_Gust_Direction',
    'cos_最大陣風風向': 'cos_Max_Gust_Direction',
    'sin_風向': 'sin_Wind_Direction',
    'cos_風向': 'cos_Wind_Direction',
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

# 提取座標欄位（格式為 "(經度, 緯度)"，全部 target）
target_columns = [col for col in df.columns if '(' in col and ')' in col]
print("所有座標點：", target_columns)

# 替換欄位名稱為英文供模型使用
df_tree = df.rename(columns=feature_mapping)

# 定義 X 與 y
# 如需加入處理後的角度特徵，可自行擴充此處特徵清單
X = df_original[list(feature_mapping.keys())]
y = df_original[target_columns]

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tree = X_train.rename(columns=feature_mapping)
X_test_tree = X_test.rename(columns=feature_mapping)

# ---------------------------
# 定義輔助函式

def get_breadth_first_path(tree_dict):
    """
    以廣度優先順序遍歷樹（字典結構），返回所有節點的 (split_feature, threshold) 規則。
    """
    path = []
    q = deque([tree_dict])
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

def extract_tree_structure(dt_model):
    """
    將 DecisionTreeRegressor 的內部樹結構轉換為字典形式。
    """
    tree = dt_model.tree_
    def recurse(node_index):
        # 如果該節點為葉節點
        if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
            return {}
        node_dict = {
            "split_feature": tree.feature[node_index],
            "threshold": tree.threshold[node_index]
        }
        left_index = tree.children_left[node_index]
        right_index = tree.children_right[node_index]
        if left_index != -1:
            node_dict["left_child"] = recurse(left_index)
        if right_index != -1:
            node_dict["right_child"] = recurse(right_index)
        return node_dict
    return recurse(0)

def assign_group_by_feature_prefix(rule_paths, threshold):
    """
    根據每個 target 的規則路徑（僅取 split_feature 部分）進行分組，
    若某前綴下 target 數超過 threshold，則用更長前綴細分。
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

# ---------------------------
# 主循環：使用 CART 決策樹進行模型訓練與規則提取

predictions = {}
grid_ids = []
rule_paths = {}  # 儲存每個 target 的廣度優先規則路徑
target_rmse = {}
target_models = {}
target_best_tree_index = {}
geo_coords = []

for target in target_columns:
    print(f"訓練與預測 {target} 的 CART 決策樹...")
    # 使用 DecisionTreeRegressor 訓練模型
    dt_model = DecisionTreeRegressor(random_state=42, max_leaf_nodes=31, max_depth=10, min_samples_split=10, min_samples_leaf=10)
    dt_model.fit(X_train_tree, y_train[target])
    y_pred = dt_model.predict(X_test_tree)
    rmse = np.sqrt(np.mean((y_test[target] - y_pred)**2))
    predictions[target] = y_pred
    target_rmse[target] = rmse
    target_models[target] = dt_model
    target_best_tree_index[target] = 0  # CART 為單棵樹，索引為 0
    print(f"{target} 的 RMSE: {rmse}")
    
    # 繪製 CART 決策樹圖
    plt.figure(figsize=(30, 18))
    plot_tree(dt_model, feature_names=list(X_train_tree.columns), filled=True)
    plt.title(f"Decision Tree for {target} (RMSE最低)")
    tree_plot_path = os.path.join(result_dir, f"tree_{target.replace(',', '_').replace(' ', '')}.png")
    plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"CART 決策樹圖已儲存至: {tree_plot_path}")
    
    # SHAP 分析 (使用 shap.TreeExplainer 適用於 scikit-learn 決策樹)
    explainer = shap.TreeExplainer(dt_model)
    shap_values = explainer.shap_values(X_test_tree)
    shap_summary_path = os.path.join(result_dir, f"shap_summary_{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_tree, show=False)
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP 特徵重要性圖已儲存至: {shap_summary_path}")
    
    shap_bar_path = os.path.join(result_dir, f"shap_bar_{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_tree, plot_type="bar", show=False)
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP 特徵影響條形圖已儲存至: {shap_bar_path}")
    
    # 提取 CART 決策樹結構（轉換為字典格式）
    tree_dict = extract_tree_structure(dt_model)
    # 提取廣度優先規則路徑
    path = get_breadth_first_path(tree_dict)
    rule_paths[target] = path
    # 將 target 本身（座標字串）作為代表經緯度
    geo_coords.append(target)
    grid_ids.append(target)

# ---------------------------
# 分群：拓展規則
# 規則：
# 1) 初步以根部規則 (split_feature) 分組
# 2) 如果某組 target 數量超過總數的 1/10，
#    則對該組 target 進一步依據廣度優先規則路徑的前綴細分分組（只使用 split_feature，不考慮 threshold）
total_targets = len(target_columns)
threshold = total_targets / 10.0

final_groups = assign_group_by_feature_prefix(rule_paths, threshold)

# 將每個群中 RMSE 最低的 target 選為群代表
group_to_targets = {}
for target, group_prefix in final_groups.items():
    group_to_targets.setdefault(group_prefix, []).append(target)
group_representative = {}
for group_prefix, targets in group_to_targets.items():
    best_target = min(targets, key=lambda t: target_rmse[t])
    group_representative[group_prefix] = best_target

# 存群代表的模型檔
for group_prefix, rep_target in group_representative.items():
    model = target_models[rep_target]
    safe_target = rep_target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    model_file = os.path.join(result_dir, f"model_{safe_target}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"群代表 {rep_target} 的模型已存至: {model_file}")

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
    count = len(targets_in_prefix)
    rep_target = group_representative[prefix]
    rep_rmse = target_rmse[rep_target]
    group_rows.append({
        "規則": prefix_str,
        "座標數": count,
        "分組標籤": label,
        "群代表座標": rep_target,
        "代表座標RMSE": rep_rmse,
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
plt.title("基於決策樹根部規則(廣度分層)分群的地理分布")
plt.colorbar(label="Group Label")
grouping_path = os.path.join(result_dir, "geo_grouping_by_root_feature_refined.png")
plt.savefig(grouping_path, dpi=300, bbox_inches="tight")
plt.close()
print("基於決策樹根部規則(廣度分層)的地理分群圖已儲存至:", grouping_path)

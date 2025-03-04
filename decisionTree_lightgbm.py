import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
# 設定與數據讀取
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = r"C:\thesis\code\Taipei_5x5\all_merged_5X5.csv"
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
        assigned_prefix = feature_path  # 預設使用完整前綴
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

# 新增：記錄每個 target 的最佳 MSE、模型物件及最佳樹索引
target_mse = {}
target_models = {}
target_best_tree_index = {}

for target in target_columns:
    print(f"訓練與預測 {target} 的決策樹...")
    train_data = lgb.Dataset(X_train_tree, label=y_train[target], categorical_feature=cat_features)
    test_data = lgb.Dataset(X_test_tree, label=y_test[target], reference=train_data, categorical_feature=cat_features)
    # 將 metric 改為 "l2" (即 MSE)
    params = {
        'objective': 'regression',
        'metric': 'l2',
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
    if "valid_0" in evals_result and "l2" in evals_result["valid_0"]:
        plt.figure(figsize=(8, 5))
        plt.plot(evals_result['valid_0']['l2'], label="Validation MSE", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.title(f"Learning Curve ({target})")
        plt.legend()
        learning_curve_path = os.path.join(result_dir, "learning_curve", f"{target.replace(',', '_').replace(' ', '')}.png")
        plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"學習曲線已儲存至: {learning_curve_path}")
    else:
        print(f"無法繪製 {target} 的學習曲線。")

    model_dict = lgb_model.dump_model()
    tree_info = model_dict["tree_info"]
    # 取每棵樹根節點的 split_gain，若不存在則設為 0
    split_gains = [tree_info[i]["tree_structure"].get("split_gain", 0) for i in range(len(tree_info))]
    best_tree_index = np.argmax(split_gains)
    # 記錄該樹對應的 MSE (僅供參考)
    target_mse[target] = evals_result["valid_0"]["l2"][best_tree_index]
    target_models[target] = lgb_model
    target_best_tree_index[target] = best_tree_index

    plt.figure(figsize=(30, 18))
    # 移除 feature_names 參數
    lgb.plot_tree(lgb_model, tree_index=best_tree_index, show_info=['split_gain'], filled=True)
    plt.title(f"Best Decision Tree for {target} (Highest split_gain)")
    tree_plot_path = os.path.join(result_dir, "tree", f"{target.replace(',', '_').replace(' ', '')}.png")
    plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"最佳決策樹圖（Highest split_gain）已儲存至: {tree_plot_path}")


    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_summary_path = os.path.join(result_dir, "shap_summary", f"{target.replace(',', '_').replace(' ', '')}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP 特徵重要性圖已儲存至: {shap_summary_path}")

    shap_bar_path = os.path.join(result_dir, "shap_bar", f"{target.replace(',', '_').replace(' ', '')}.png")
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
    geo_coords.append(target)
    grid_ids.append(target)

# ------------------------
# 分群規則：
total_targets = len(target_columns)
threshold = total_targets / 10.0

final_groups = assign_group_by_feature_prefix(rule_paths, threshold)

# 將每個群中 MSE 最低的座標選為群代表
group_to_targets = {}
for target, group_prefix in final_groups.items():
    group_to_targets.setdefault(group_prefix, []).append(target)

group_representative = {}
for group_prefix, targets in group_to_targets.items():
    best_target = min(targets, key=lambda t: target_mse[t])
    group_representative[group_prefix] = best_target

# 將群代表的模型視覺化，同時標示出該群的規則
reverse_mapping = {v: k for k, v in feature_mapping.items()}
for group_prefix, rep_target in group_representative.items():
    model = target_models[rep_target]
    best_tree_index = target_best_tree_index[rep_target]
    
    # 只使用群規則前綴 (group_prefix)，組成解釋文字
    rule_features = []
    for feature_index in group_prefix:
        if feature_index < len(X_train_tree.columns):
            feature_eng = list(X_train_tree.columns)[feature_index]
        else:
            feature_eng = str(feature_index)
        feature_ch = reverse_mapping.get(feature_eng, feature_eng)
        rule_features.append(feature_ch)
    rule_text = f"群代表座標: {rep_target}\n群規則: " + "; ".join(rule_features)
    
    # 繪製決策樹圖，並在圖上標示解釋文字
    plt.figure(figsize=(30, 18))
    lgb.plot_tree(model, tree_index=best_tree_index, show_info=['split_gain'])
    plt.suptitle(rule_text, fontsize=20)
    safe_target = rep_target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    rep_tree_plot_path = os.path.join(result_dir, "group_tree", f"group_representative_tree_{safe_target}.png")
    plt.savefig(rep_tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"群代表 {rep_target} 的決策樹圖已存至: {rep_tree_plot_path}")



# 建立 target -> 分組標籤對照表：將唯一前綴映射到數值標籤
unique_prefixes = {v for v in final_groups.values()}
prefix_to_label = {prefix: idx for idx, prefix in enumerate(unique_prefixes)}
group_labels = {target: prefix_to_label[final_groups[target]] for target in final_groups}

# 輸出分群結果到 CSV：僅顯示分組時使用的特徵名稱 (中文)
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
    rep_mse = target_mse[rep_target]
    group_mse = np.mean([target_mse[t] for t in targets_in_prefix])
    overall_mse = np.mean(list(target_mse.values()))
    group_rows.append({
        "規則": prefix_str,
        "座標數": count,
        "分組標籤": label,
        "群代表座標": rep_target,
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
plt.title("基於決策樹根部規則(廣度分層)分群的地理分布")
plt.colorbar(label="Group Label")
grouping_path = os.path.join(result_dir, "geo_grouping_by_root_feature_refined.png")
plt.savefig(grouping_path, dpi=300, bbox_inches="tight")
plt.close()
print("基於決策樹根部規則(廣度分層)的地理分群圖已儲存至:", grouping_path)


# 建立儲存群代表決策樹的子目錄
group_tree_dir = os.path.join(result_dir, "group_tree")
os.makedirs(group_tree_dir, exist_ok=True)

# 對每個群代表進行繪圖與標示群規則
for prefix, rep_target in group_representative.items():
    # 從 group_rows 中找出對應群規則字串
    rule_str = ""
    for row in group_rows:
        if row["群代表座標"] == rep_target:
            rule_str = row["規則"]
            break
    # 取得該群代表的決策樹模型
    model = target_models[rep_target]
    plt.figure(figsize=(30, 18))
    # 繪製決策樹，移除 feature_names 參數
    lgb.plot_tree(model, tree_index=target_best_tree_index[rep_target], show_info=['split_gain'], filled=True)
    # 在圖上標題處加入群代表與群規則說明
    plt.suptitle(f"群代表: {rep_target}\n群規則: {rule_str}", fontsize=20)
    safe_target = rep_target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    group_tree_path = os.path.join(group_tree_dir, f"group_representative_tree_{safe_target}.png")
    plt.savefig(group_tree_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"群代表 {rep_target} 的標註規則決策樹圖已存至: {group_tree_path}")

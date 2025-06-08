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
sub_dirs = ["learning_curve", "tree", "shap_summary", "shap_bar", "model", "group_tree", "individual_target_models"]
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

# 解析座標字串函式 (如果尚未在全域定義，則移至此處或確保可訪問)
def parse_coord_string(coord_str):
    coord_str = coord_str.strip("() ")
    lon_str, lat_str = coord_str.split(",")
    return float(lon_str), float(lat_str)

# --- 新增：篩選最中心的400個座標點 ---
if target_columns:
    parsed_coords = []
    for tc_str in target_columns:
        try:
            lon, lat = parse_coord_string(tc_str)
            parsed_coords.append({'name': tc_str, 'lon': lon, 'lat': lat})
        except ValueError:
            print(f"警告：無法解析座標字串 '{tc_str}'，將跳過此座標。")
            continue
    
    if parsed_coords:
        coords_df = pd.DataFrame(parsed_coords)
        
        # 計算中心點
        center_lon = coords_df['lon'].mean()
        center_lat = coords_df['lat'].mean()
        print(f"計算出的地理中心點: (經度: {center_lon:.4f}, 緯度: {center_lat:.4f})")

        # 計算每個點到中心點的距離 (歐幾里得距離的平方，避免開根號，排序結果一致)
        coords_df['distance_sq'] = (coords_df['lon'] - center_lon)**2 + (coords_df['lat'] - center_lat)**2
        
        # 排序並選取最近的N個點
        num_points_to_select = 400
        if len(coords_df) > num_points_to_select:
            selected_coords_df = coords_df.nsmallest(num_points_to_select, 'distance_sq')
            print(f"已篩選出最靠近中心的 {num_points_to_select} 個座標點。")
        else:
            selected_coords_df = coords_df.copy()
            print(f"座標點總數 ({len(coords_df)}) 少於或等於 {num_points_to_select}，已選取所有座標點。")

        # 更新 target_columns
        original_target_count = len(target_columns)
        target_columns = sorted(list(selected_coords_df['name']))
        print(f"座標點數量從 {original_target_count} 更新為 {len(target_columns)}。")

        # 繪製篩選後的400個點
        plt.figure(figsize=(10, 8))
        plt.scatter(selected_coords_df['lon'], selected_coords_df['lat'], s=10, alpha=0.7, label=f"篩選出的 {len(target_columns)} 個中心點")
        plt.scatter(center_lon, center_lat, color='red', marker='x', s=100, label="地理中心")
        plt.xlabel("經度 (Longitude)")
        plt.ylabel("緯度 (Latitude)")
        plt.title(f"篩選出的 {len(target_columns)} 個中心座標點地理分佈")
        plt.legend()
        plt.grid(True)
        plt.ticklabel_format(useOffset=False, style='plain', axis='both')
        selected_points_map_path = os.path.join(result_dir, "selected_central_400_points_map.png")
        plt.savefig(selected_points_map_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"篩選後的中心座標點地圖已儲存至: {selected_points_map_path}")
    else:
        print("警告：沒有可用的有效座標點進行篩選。")
else:
    print("警告：target_columns 為空，無法進行座標篩選。")
# --- 篩選結束 ---


# 替換 DataFrame 欄位名稱為英文供 LightGBM 使用
df_tree = df.rename(columns=feature_mapping)

# 定義 X 與 y (y 現在會使用篩選後的 target_columns)
X = df_original[list(feature_mapping.keys())]
y = df_original[target_columns] # 確保 y 只包含篩選後的目標

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tree = X_train.rename(columns=feature_mapping)
X_test_tree = X_test.rename(columns=feature_mapping)

cat_features = ['Holiday']
# 將 feature_names_english 的定義移到此處，確保在後續的規則統計和分群邏輯中可用
feature_names_english = list(X_train_tree.columns) 

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

# 將 format_rule_to_string 函數定義移到此處
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

# ---------------------------
# 主循環：對每個 target 訓練模型、提取規則
predictions = {}
grid_ids = []
geo_coords = []
rule_paths = {}    # 舊的規則路徑儲存，新方法直接從樹結構提取特定規則

# 新增：記錄每個 target 的最佳 MAE、模型物件及最佳樹索引
target_mae = {}
target_mse = {}
target_models = {}
target_best_tree_index = {}
print(f"訓練單獨模型...")
individual_models_dir = os.path.join(result_dir, "individual_target_models") 

for target in target_columns: # 此迴圈現在只會遍歷篩選後的 target_columns
    print(f"\n處理目標網格: {target}")
    # 產生模型檔案路徑 (確保檔案名稱合法)
    safe_target_filename = "".join(c if c.isalnum() else "_" for c in target)
    model_path = os.path.join(individual_models_dir, f"model_{safe_target_filename}.txt")

    if os.path.exists(model_path):
        print(f"  載入已存在的模型: {model_path}")
        lgb_model = lgb.Booster(model_file=model_path)
        # 載入模型後，重新預測以計算評估指標
        y_pred = lgb_model.predict(X_test_tree, num_iteration=lgb_model.current_iteration()) # 使用 current_iteration() 或 best_iteration (如果可用)
        
        # 更新相關字典
        target_models[target] = lgb_model
        target_mae[target] = mean_absolute_error(y_test[target], y_pred)
        target_mse[target] = mean_squared_error(y_test[target], y_pred)
        
        model_dict_loaded = lgb_model.dump_model()
        tree_info_loaded = model_dict_loaded["tree_info"]
        # 假設最佳樹是基於 split_gain 最高的 (通常 LightGBM 會儲存所有樹)
        # 如果只存了 best_iteration 對應的樹，則 tree_info 可能只有一個元素
        # 這裡我們假設需要從多棵樹中選 split_gain 最高的
        split_gains_loaded = [tree_info_loaded[i]["tree_structure"].get("split_gain", 0) for i in range(len(tree_info_loaded))]
        if not split_gains_loaded: # 以防萬一 tree_info 為空或沒有 split_gain
            print(f"警告: 載入的模型 {target} 沒有有效的 split_gain。將使用索引 0。")
            target_best_tree_index[target] = 0
        else:
            target_best_tree_index[target] = np.argmax(split_gains_loaded)
        
        print(f"  模型 {target} MAE: {target_mae[target]:.4f}, MSE: {target_mse[target]:.4f}")
        # 不需要繪製學習曲線，因為模型已訓練
        # SHAP 和 tree plot 仍可基於載入模型生成

    else:
        print(f"  為 {target} 訓練新模型...")
        train_data = lgb.Dataset(X_train_tree, label=y_train[target], categorical_feature=cat_features)
        test_data = lgb.Dataset(X_test_tree, label=y_test[target], reference=train_data, categorical_feature=cat_features)
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
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=100), # 調整 verbose
                       lgb.record_evaluation(evals_result),
                       lgb.log_evaluation(100)]
        )
        y_pred = lgb_model.predict(X_test_tree, num_iteration=lgb_model.best_iteration)
        
        # 儲存模型
        lgb_model.save_model(model_path)
        print(f"  模型已儲存至: {model_path}")

        predictions[target] = y_pred # predictions 字典似乎沒有在後續使用，但保留以防萬一
        target_models[target] = lgb_model
        target_mae[target] = mean_absolute_error(y_test[target], y_pred)
        target_mse[target] = mean_squared_error(y_test[target], y_pred)

        model_dict_trained = lgb_model.dump_model()
        tree_info_trained = model_dict_trained["tree_info"]
        split_gains_trained = [tree_info_trained[i]["tree_structure"].get("split_gain", 0) for i in range(len(tree_info_trained))]
        if not split_gains_trained:
             print(f"警告: 新訓練的模型 {target} 沒有有效的 split_gain。將使用索引 0。")
             target_best_tree_index[target] = 0
        else:
            target_best_tree_index[target] = np.argmax(split_gains_trained)

        if "valid_0" in evals_result and "l2" in evals_result["valid_0"]:
            plt.figure(figsize=(8, 5))
            plt.plot(evals_result['valid_0']['l2'], label="Validation MSE", color="blue")
            plt.xlabel("Iterations")
            plt.ylabel("Error")
            plt.title(f"Learning Curve ({target})")
            plt.legend()
            learning_curve_path = os.path.join(result_dir, "learning_curve", f"{safe_target_filename}.png")
            plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            print(f"無法繪製 {target} 的學習曲線。")

    # 以下部分對載入或新訓練的模型都執行
    # 檢查 target_models 是否包含當前 target，以防在篩選後某些 target 沒有模型被載入或訓練
    if target not in target_models:
        print(f"警告: 目標 {target} 未在 target_models 中找到，可能在之前的步驟中被跳過。跳過此目標的後續處理。")
        # 確保 geo_coords 和 grid_ids 即使在跳過時也可能需要更新，取決於它們的用途
        # 如果它們嚴格對應已處理的模型，則此處不應添加
        # all_target_rules_info 的填充邏輯在後續會處理這種情況
        continue # 跳到下一個 target

    current_model_dict = target_models[target].dump_model()
    current_tree_info = current_model_dict["tree_info"]
    current_best_tree_idx = target_best_tree_index[target]

    # 確保 current_best_tree_idx 在 current_tree_info 的有效範圍內
    if current_best_tree_idx >= len(current_tree_info):
        print(f"警告: 目標 {target} 的 best_tree_index ({current_best_tree_idx}) 超出 tree_info 範圍 ({len(current_tree_info)})。將使用索引 0。")
        current_best_tree_idx = 0
        target_best_tree_index[target] = 0 # 更新儲存的索引
        if not current_tree_info: # 如果 tree_info 為空，則無法繼續繪圖或分析
            print(f"錯誤: 目標 {target} 的 tree_info 為空，跳過此目標的後續處理。")
            geo_coords.append(target) 
            grid_ids.append(target)
            # 為 all_target_rules_info 設置預設值 -> 這部分由後續的 all_target_rules_info 填充邏輯處理
            continue


    plt.figure(figsize=(30, 18))
    lgb.plot_tree(target_models[target], tree_index=current_best_tree_idx, show_info=['split_gain', 'data_count'])
    plt.title(f"Best Decision Tree for {target} (Highest split_gain)")
    tree_plot_path = os.path.join(result_dir, "tree", f"{safe_target_filename}.png")
    plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    plt.close()

    explainer = shap.TreeExplainer(target_models[target])
    shap_values = explainer.shap_values(X_test_tree) # 注意：SHAP 值應基於 X_test_tree (英文特徵名)
    shap_summary_path = os.path.join(result_dir, "shap_summary", f"{safe_target_filename}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_tree, show=False) # X_test_tree
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    shap_bar_path = os.path.join(result_dir, "shap_bar", f"{safe_target_filename}.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_tree, plot_type="bar", show=False) # X_test_tree
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 提取規則的部分依賴於 target_models 和 target_best_tree_index，這些已經被正確設定
    geo_coords.append(target) 
    grid_ids.append(target)
    
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
# coords = [parse_coord_string(coord) for coord in target_columns] # target_columns 已更新
# lons = np.array([coord[0] for coord in coords])
# lats = np.array([coord[1] for coord in coords])
# 上述提取 lons, lats 的部分如果僅用於共享模型數據準備，則應在共享模型部分重新計算或傳遞篩選後的 lons, lats

# 重塑訓練數據
n_samples_train = X_train.shape[0]
n_targets = len(target_columns) # n_targets 現在會是 400 (或實際篩選數量)
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
shap_values = explainer.shap_values(X_test_expanded) # X_test_expanded 的特徵名應為英文

# 為每個網格生成 SHAP 圖
for i, target in enumerate(target_columns): # target_columns 已更新
    safe_target = target.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    # idx 的計算需要基於更新後的 n_targets (即 len(target_columns))
    # n_samples_test 保持不變
    idx_start = i * n_samples_test
    idx_end = (i + 1) * n_samples_test
    
    # SHAP 摘要圖（點圖）
    plt.figure(figsize=(10, 8))
    # 使用 X_test_expanded.iloc[idx_start:idx_end] 確保特徵名一致
    shap.summary_plot(shap_values[idx_start:idx_end], X_test_expanded.iloc[idx_start:idx_end], plot_type='dot', show=False)
    plt.title(f'共享模型 SHAP 摘要圖 ({target})')
    plt.savefig(os.path.join(shared_result_dir, 'shap_summary', f'shared_model_{safe_target}.png'), dpi=300)
    plt.close()
    
    # SHAP 特徵重要性圖（條形圖）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[idx_start:idx_end], X_test_expanded.iloc[idx_start:idx_end], plot_type='bar', show=False)
    plt.title(f'共享模型 SHAP 特徵重要性 ({target})')
    plt.savefig(os.path.join(shared_result_dir, 'shap_bar', f'shared_model_{safe_target}.png'), dpi=300)
    plt.close()

# 生成平均 SHAP 圖
# mean_shap_values 的 reshape 也需要使用更新後的 n_targets
mean_shap_values = np.mean(shap_values.reshape(n_targets, n_samples_test, -1), axis=0)
X_test_for_shap = X_test_expanded.iloc[:n_samples_test] # 保持不變，取第一批樣本的特徵

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
for i, target in enumerate(target_columns): # target_columns 已更新
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
# 新增：規則得分統計邏輯開始

# 輔助函數：從樹結構中提取所有分裂節點，按增益排序
def extract_all_split_nodes_sorted(tree_structure, feature_names_eng_list_param, cat_features_eng_list_param): # 參數名稱已修改以區分
    """
    從樹結構中提取所有有效的分裂節點資訊，並按 split_gain 降序排序。
    返回列表，每個元素為 {'rule': (特徵索引, 閾值), 'gain': gain}
    """
    nodes_info = []
    q = deque()
    if tree_structure:
        q.append(tree_structure)

    while q:
        node = q.popleft()
        if node and "split_feature" in node and node.get("split_gain", 0) > 0:
            feat_idx = node["split_feature"]
            threshold = node["threshold"]
            gain = node["split_gain"]
            
            # 使用傳入的參數 feature_names_eng_list_param
            if feat_idx >= len(feature_names_eng_list_param):
                # print(f"警告: 特徵索引 {feat_idx} 在提取規則時超出範圍。")
                continue
            
            nodes_info.append({
                'rule': (feat_idx, threshold), 
                'gain': gain
            })

            if "left_child" in node and node["left_child"]:
                q.append(node["left_child"])
            if "right_child" in node and node["right_child"]:
                q.append(node["right_child"])
                
    nodes_info.sort(key=lambda x: x['gain'], reverse=True)
    return nodes_info

# 全域列表，用於收集所有目標點的原始評分規則
all_scored_rules_raw = []
# feature_names_english 已在前面定義 (全域)
# cat_features 已在前面定義 (全域)

MAX_RANK_SCORE_VALUE = 5 # 最高排名得分值 (例如5分給第一名)
NUM_RULES_TO_SCORE = 5   # 為每個目標點評分的前N條規則

print("開始為每個目標點的規則進行評分...")
for target in target_columns:
    if target not in target_models or target_best_tree_index.get(target) is None:
        # print(f"目標 {target} 沒有模型或最佳樹索引，跳過規則評分。")
        continue

    model_dump = target_models[target].dump_model()
    best_tree_idx = target_best_tree_index[target]

    if not model_dump["tree_info"] or best_tree_idx >= len(model_dump["tree_info"]):
        # print(f"目標 {target} 的樹信息不足，跳過規則評分。")
        continue
    
    best_tree_structure = model_dump["tree_info"][best_tree_idx]["tree_structure"]
    # sorted_split_nodes 包含此目標點最佳樹中所有分裂節點，按 split_gain 降序排序
    # 每個元素是 {'rule': (特徵索引, 閾值), 'gain': gain}
    sorted_split_nodes = extract_all_split_nodes_sorted(best_tree_structure, 
                                                        feature_names_english, 
                                                        cat_features) 
    
    # 只考慮此目標點貢獻最大的前 NUM_RULES_TO_SCORE 條規則
    for rank, node_info in enumerate(sorted_split_nodes[:NUM_RULES_TO_SCORE], start=1):
        rule_tuple = node_info['rule']
        original_gain_value = node_info['gain'] # 原始 split_gain
        
        feat_idx, thresh_val = rule_tuple
        
        if feat_idx >= len(feature_names_english): 
            # print(f"警告: 特徵索引 {feat_idx} 在評分時超出範圍 (目標: {target}, 規則: {rule_tuple})。")
            continue
        
        feature_name = feature_names_english[feat_idx]
        is_categorical = feature_name in cat_features
        
        # 步驟1: 初步評分 - 計算排名得分
        # rank 1 (split_gain最高) -> MAX_RANK_SCORE_VALUE 分
        # rank 2 -> MAX_RANK_SCORE_VALUE - 1 分, ..., rank N -> 1 分
        rank_based_score = MAX_RANK_SCORE_VALUE - rank + 1
        
        rule_dict = {
            'target': target,
            'rule_tuple': rule_tuple,
            'score': rank_based_score, # 此處的 'score' 是基於排名的初步分數
            'rank': rank,
            'is_categorical': is_categorical,
            'original_gain': original_gain_value # 保留原始gain以供參考
        }
        all_scored_rules_raw.append(rule_dict)

# 檢查是否有有效的規則被提取
if not all_scored_rules_raw:
    print("警告: 沒有有效的規則被提取。請檢查模型和樹的結構。")
else:
    print(f"成功提取 {len(all_scored_rules_raw)} 條規則。")

# 輔助函數：合併相似的評分規則並匯總統計
def generate_merged_rule_score_statistics(scored_rules_raw_list, 
                                          feature_names_eng_list_param, # 參數名稱已修改以區分
                                          cat_features_eng_list_param,  # 參數名稱已修改以區分
                                          threshold_similarity_dist):
    # merged_stats 的結構:
    # Key: 代表性規則元組 (feat_idx, thresh_val)
    # Value: {'score': 累計總分數, 'targets': set(貢獻分數的目標點名稱)}
    merged_stats = {}  # Key: representative_rule_tuple, Value: {'score': sum_score, 'targets': set_of_targets}
    representative_rules_tuples_list = [] # 用於存儲已經確立的代表性規則元組，方便快速查找

    for scored_item in scored_rules_raw_list:
        current_rule_tuple = scored_item['rule_tuple']
        current_score = scored_item['score'] # 這是來自 all_scored_rules_raw 的 rank_based_score
        current_target = scored_item['target']
        
        feat_idx, thresh_val = current_rule_tuple
        
        # 使用傳入的參數 feature_names_eng_list_param
        if feat_idx >= len(feature_names_eng_list_param):
            # print(f"警告: 特徵索引 {feat_idx} 在評分時超出範圍 (目標: {target}, 規則: {rule_tuple})。")
            continue
        
        feature_name = feature_names_eng_list_param[feat_idx]
        is_categorical = feature_name in cat_features_eng_list_param
        
        matched_representative_tuple = None
        for rep_rule_tuple in representative_rules_tuples_list:
            rep_feat_idx, rep_thresh_val = rep_rule_tuple
            
            # 使用傳入的參數 feature_names_eng_list_param
            if rep_feat_idx >= len(feature_names_eng_list_param): 
                continue
            rep_feature_name = feature_names_eng_list_param[rep_feat_idx]
            is_rep_categorical = rep_feature_name in cat_features_eng_list_param # 使用傳入的參數

            # 步驟2: 合併相似規則並匯總分數
            if feat_idx == rep_feat_idx: # 特徵必須相同
                if is_categorical and is_rep_categorical:
                    # 類別特徵：閾值字串必須完全相同
                    if str(thresh_val) == str(rep_thresh_val):
                        matched_representative_tuple = rep_rule_tuple
                        break
                else:
                    # 數值特徵：閾值差異在允許範圍內
                    if abs(thresh_val - rep_thresh_val) <= threshold_similarity_dist:
                        matched_representative_tuple = rep_rule_tuple
                        break
        
        if matched_representative_tuple is not None:
            # 如果找到匹配的代表規則，則將當前規則的初步分數累加到代表規則的總分數上
            if matched_representative_tuple in merged_stats:
                merged_stats[matched_representative_tuple]['score'] += current_score
                merged_stats[matched_representative_tuple]['targets'].add(current_target)
            else:
                # 理論上 matched_representative_tuple 應該已經在 merged_stats 中，除非列表管理有誤
                # 為保險起見，如果不在，則初始化 (雖然正常情況下不應進入此分支)
                merged_stats[matched_representative_tuple] = {'score': current_score, 'targets': {current_target}}
        else:
            # 否則，將當前規則作為新的代表規則，其初步分數作為初始總分數
            representative_rules_tuples_list.append(current_rule_tuple)
            merged_stats[current_rule_tuple] = {'score': current_score, 'targets': {current_target}}

    return merged_stats

# 執行合併與統計
# merged_rule_scores 將包含每個代表性規則的最終累計分數
merged_rule_scores = generate_merged_rule_score_statistics(all_scored_rules_raw,
                                                           feature_names_english,
                                                           cat_features, 
                                                           1.5)

# 準備輸出到 Excel
output_scored_rules_data = []
for rule_tuple, stats in merged_rule_scores.items():
    rule_str = format_rule_to_string(rule_tuple, feature_names_english, reverse_mapping, cat_features)
    targets_str = ", ".join(sorted(list(stats['targets'])))
    # 步驟3: 最終輸出 - stats['score'] 即為該規則的最終累計分數
    output_scored_rules_data.append({
        '規則': rule_str,
        '分數': stats['score'], # 欄位名稱改為 "分數"
        '目標數量': len(stats['targets']), # 此欄位可選，如果不需要可以移除
        '貢獻分數的座標點列表': targets_str # 欄位名稱更新
    })

output_scored_rules_df = pd.DataFrame(output_scored_rules_data)
# 按分數降序排序
output_scored_rules_df = output_scored_rules_df.sort_values(by="分數", ascending=False)

# 更新檔名以反映計分方式和內容
scored_rules_csv_path = os.path.join(shared_result_dir, "ranked_rule_score_statistics.csv")
scored_rules_excel_path = os.path.join(shared_result_dir, "ranked_rule_score_statistics.xlsx")

output_scored_rules_df.to_csv(scored_rules_csv_path, index=False, encoding='utf-8-sig')
print(f"排名式規則得分統計結果已儲存至 CSV: {scored_rules_csv_path}")
try:
    output_scored_rules_df.to_excel(scored_rules_excel_path, index=False, engine='openpyxl')
    print(f"排名式規則得分統計結果已儲存至 Excel: {scored_rules_excel_path}")
except ImportError:
    print("警告: 未安裝 'openpyxl'。Excel 檔案未儲存。請執行 'pip install openpyxl'")

# 新分群邏輯開始

# 輔助函數：從節點字典中提取規則 (特徵索引, 閾值), 增益及節點本身
def get_rule_and_gain_node(node_dict):
    """從LGBM樹模型節點字典中提取分裂規則、增益及節點本身。"""
    if node_dict and "split_feature" in node_dict and node_dict.get("split_gain", 0) > 0: # 確保是有效分裂
        rule = (node_dict["split_feature"], node_dict["threshold"])
        gain = node_dict["split_gain"]
        return rule, gain, node_dict # Return node_dict as well
    return None, -float('inf'), None # Consistent return for invalid/leaf nodes

# 新增輔助函數：獲取指定深度按增益排序的節點資訊
def get_nodes_info_at_depth_sorted_by_gain(tree_structure, target_depth):
    """
    遍歷樹到目標深度，收集該深度的分裂節點信息 (規則、增益、節點本身)，
    並按 split_gain 降序排序返回。
    深度為 1-indexed (根節點為深度 1)。
    返回列表，每個元素為 {'rule': rule_tuple, 'gain': gain, 'node_dict': node_dict}
    """
    nodes_info_list = [] 
    
    queue = deque()
    if tree_structure: # 確保根節點存在
        queue.append({'node_dict': tree_structure, 'current_depth': 1})

    while queue:
        item = queue.popleft()
        node_dict_current = item['node_dict']
        current_depth = item['current_depth']

        if node_dict_current is None:
            continue

        if current_depth == target_depth:
            rule, gain, node_obj = get_rule_and_gain_node(node_dict_current) 
            if rule: # 如果是有效的帶增益的分裂規則
                nodes_info_list.append({'rule': rule, 'gain': gain, 'node_dict': node_obj})
        
        elif current_depth < target_depth:
            left_child = node_dict_current.get("left_child")
            if left_child:
                queue.append({'node_dict': left_child, 'current_depth': current_depth + 1})
            
            right_child = node_dict_current.get("right_child")
            if right_child:
                queue.append({'node_dict': right_child, 'current_depth': current_depth + 1})
                
    nodes_info_list.sort(key=lambda x: x['gain'], reverse=True)
    
    return nodes_info_list

# 新增輔助函數：獲取父節點的子節點資訊，按增益排序
def get_children_info_sorted_by_gain(parent_node_dict):
    """
    獲取父節點的子節點中，有效的子節點資訊 (規則、增益、節點字典)，按增益降序排序。
    返回列表: [{'rule': ..., 'gain': ..., 'node_dict': ...}, ...]
    """
    children_info = []
    if not parent_node_dict:
        return children_info

    left_child_node = parent_node_dict.get("left_child")
    if left_child_node:
        rule, gain, node = get_rule_and_gain_node(left_child_node)
        if rule: # Only add if it's a valid split node
            children_info.append({'rule': rule, 'gain': gain, 'node_dict': node})

    right_child_node = parent_node_dict.get("right_child")
    if right_child_node:
        rule, gain, node = get_rule_and_gain_node(right_child_node)
        if rule: # Only add if it's a valid split node
            children_info.append({'rule': rule, 'gain': gain, 'node_dict': node})
    
    children_info.sort(key=lambda x: x['gain'], reverse=True)
    return children_info


# 輔助函數：為單一目標提取 R1, R2, R3 規則及其節點資訊 (新定義)
all_target_rules_info = {} 
print("提取各目標點的 R1, R2, R3 規則及節點資訊...")

empty_node_info = {'rule': None, 'gain': -float('inf'), 'node_dict': None}

for target in target_columns: 
    current_target_info = {
        'r1_root': empty_node_info.copy(),
        'r2_d2top': empty_node_info.copy(),
        'r3_d2second': empty_node_info.copy()
    }

    if target not in target_models: 
        all_target_rules_info[target] = current_target_info
        continue
    model_dump = target_models[target].dump_model()

    if not model_dump["tree_info"] or target_best_tree_index[target] >= len(model_dump["tree_info"]):
        all_target_rules_info[target] = current_target_info
        continue
    
    best_tree_structure = model_dump["tree_info"][target_best_tree_index[target]]["tree_structure"]

    r1_rule, r1_gain, r1_node = get_rule_and_gain_node(best_tree_structure)
    if r1_rule:
        current_target_info['r1_root'] = {'rule': r1_rule, 'gain': r1_gain, 'node_dict': r1_node}
    # else it remains empty_node_info
        
    depth2_nodes_info_sorted = get_nodes_info_at_depth_sorted_by_gain(best_tree_structure, 2)
    
    if len(depth2_nodes_info_sorted) > 0:
        current_target_info['r2_d2top'] = depth2_nodes_info_sorted[0]
    
    if len(depth2_nodes_info_sorted) > 1:
        current_target_info['r3_d2second'] = depth2_nodes_info_sorted[1]
    
    all_target_rules_info[target] = current_target_info

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
initial_targets = list(target_columns) # target_columns 已更新
# 每個元素是 (targets_list, defining_rules_dict)
# defining_rules_dict 結構: {'r1_root': rule_tuple, 'r2_depth2_top_gain': rule_tuple, 'r3_depth2_second_gain': rule_tuple}
processing_groups = [(initial_targets, {})] 
final_eight_groups_details = []

rule_keys_for_splitting = ['r1_root', 'r2_d2top', 'r3_d2second'] # Updated keys
print("開始分層分群...")
for i, rule_key_base in enumerate(rule_keys_for_splitting): # Use rule_key_base
    print(f"  處理分群層級: {rule_key_base} (第 {i+1} 層)")
    next_level_processing_groups = []
    for current_group_targets, path_rules_so_far in processing_groups:
        if not current_group_targets: # 如果當前群組已空，直接帶入下一層
            # 確保即使群組為空，也為其分配一個定義規則（例如 None）以保持結構
            next_level_processing_groups.append(([], {**path_rules_so_far, rule_key_base: None}))
            next_level_processing_groups.append(([], {**path_rules_so_far, rule_key_base: None}))
            continue

        # 1. 提取當前群組中所有目標在此分裂層級的規則 (rule tuple)
        rules_at_this_level_for_group = {
            t: all_target_rules_info[t][rule_key_base]['rule'] 
            for t in current_group_targets if t in all_target_rules_info and all_target_rules_info[t][rule_key_base]['rule'] is not None
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
                t: all_target_rules_info[t][rule_key_base]['rule'] # Ensure 'rule' is extracted
                for t in group_B_targets_candidates 
                if t in all_target_rules_info and all_target_rules_info[t][rule_key_base]['rule'] is not None
            }
            merged_counts_B, _ = merge_individual_rules_and_count( 
                rules_for_group_B_candidates, cat_features, 1.5, feature_names_english
            )
            defining_rule_B = find_most_frequent_rule(merged_counts_B)


        # 添加到下一層處理列表
        next_level_processing_groups.append((group_A_targets, {**path_rules_so_far, rule_key_base: defining_rule_A}))
        next_level_processing_groups.append((group_B_targets_candidates, {**path_rules_so_far, rule_key_base: defining_rule_B}))

    processing_groups = next_level_processing_groups

final_eight_groups_details = processing_groups
print(f"分群完成，共形成 {len(final_eight_groups_details)} 個最終群組。")

# 整理並匯出分群結果
output_data = []
for i, (targets_in_group, group_rules_dict) in enumerate(final_eight_groups_details):
    # 確保所有規則鍵都存在
    r1_str = format_rule_to_string(group_rules_dict.get('r1_root'), feature_names_english, reverse_mapping, cat_features)
    r2_str = format_rule_to_string(group_rules_dict.get('r2_d2top'), feature_names_english, reverse_mapping, cat_features)
    r3_str = format_rule_to_string(group_rules_dict.get('r3_d2second'), feature_names_english, reverse_mapping, cat_features)
    
    # --- 新增欄位提取邏輯 ---
    # R2 子節點規則
    r2_child1_rules_for_group_list = []
    r2_child2_rules_for_group_list = []
    for t_idx, t in enumerate(targets_in_group): 
        if t in all_target_rules_info:
            r2_node_dict_for_target = all_target_rules_info[t]['r2_d2top']['node_dict']
            if r2_node_dict_for_target:
                children_of_r2 = get_children_info_sorted_by_gain(r2_node_dict_for_target)
                if len(children_of_r2) > 0 and children_of_r2[0]['rule']:
                    r2_child1_rules_for_group_list.append(children_of_r2[0]['rule'])
                if len(children_of_r2) > 1 and children_of_r2[1]['rule']:
                    r2_child2_rules_for_group_list.append(children_of_r2[1]['rule'])
    
    merged_r2_child1_counts, _ = merge_individual_rules_and_count(
        {idx: r for idx, r in enumerate(r2_child1_rules_for_group_list)}, cat_features, 1.5, feature_names_english
    )
    dominant_r2_child1_rule = find_most_frequent_rule(merged_r2_child1_counts)
    r2_child1_str = format_rule_to_string(dominant_r2_child1_rule, feature_names_english, reverse_mapping, cat_features)

    merged_r2_child2_counts, _ = merge_individual_rules_and_count(
        {idx: r for idx, r in enumerate(r2_child2_rules_for_group_list)}, cat_features, 1.5, feature_names_english
    )
    dominant_r2_child2_rule = find_most_frequent_rule(merged_r2_child2_counts)
    r2_child2_str = format_rule_to_string(dominant_r2_child2_rule, feature_names_english, reverse_mapping, cat_features)

    # R3 子節點規則
    r3_child1_rules_for_group_list = []
    r3_child2_rules_for_group_list = []
    for t_idx, t in enumerate(targets_in_group): 
        if t in all_target_rules_info:
            r3_node_dict_for_target = all_target_rules_info[t]['r3_d2second']['node_dict']
            if r3_node_dict_for_target:
                children_of_r3 = get_children_info_sorted_by_gain(r3_node_dict_for_target)
                if len(children_of_r3) > 0 and children_of_r3[0]['rule']:
                    r3_child1_rules_for_group_list.append(children_of_r3[0]['rule'])
                if len(children_of_r3) > 1 and children_of_r3[1]['rule']:
                    r3_child2_rules_for_group_list.append(children_of_r3[1]['rule'])

    merged_r3_child1_counts, _ = merge_individual_rules_and_count(
        {idx: r for idx, r in enumerate(r3_child1_rules_for_group_list)}, cat_features, 1.5, feature_names_english
    )
    dominant_r3_child1_rule = find_most_frequent_rule(merged_r3_child1_counts)
    r3_child1_str = format_rule_to_string(dominant_r3_child1_rule, feature_names_english, reverse_mapping, cat_features)

    merged_r3_child2_counts, _ = merge_individual_rules_and_count(
        {idx: r for idx, r in enumerate(r3_child2_rules_for_group_list)}, cat_features, 1.5, feature_names_english
    )
    dominant_r3_child2_rule = find_most_frequent_rule(merged_r3_child2_counts)
    r3_child2_str = format_rule_to_string(dominant_r3_child2_rule, feature_names_english, reverse_mapping, cat_features)
    
    output_data.append({
        "組別ID": i + 1,
        "規則1 (根節點)": r1_str,
        "規則2 (第二層最高增益)": r2_str,
        "規則3 (第二層次高增益)": r3_str,
        "R2節點子節點-最高增益規則": r2_child1_str,
        "R2節點子節點-次高增益規則": r2_child2_str,
        "R3節點子節點-最高增益規則": r3_child1_str,
        "R3節點子節點-次高增益規則": r3_child2_str,
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


# target_columns 已經是排序且唯一的列表 (並且是篩選後的)
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
    '#1f77b4', '#aec7e8',  # R1_typeA 分支 -> 子分支1 (藍色系) -> 組別 0, 1
    '#2ca02c', '#98df8a',  # R1_typeA 分支 -> 子分支2 (綠色系) -> 組別 2, 3
    '#ff7f0e', '#ffbb78',  # R1_typeB 分支 -> 子分支1 (橙色系) -> 組別 4, 5
    '#d62728', '#ff9896'   # R1_typeB 分支 -> 子分支2 (紅色系) -> 組別 6, 7
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

print("所有處理完成。")
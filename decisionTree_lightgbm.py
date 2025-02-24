import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

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
df['sin_wind_direction'] = np.sin(np.deg2rad(df['風向']))            # 角度轉為sin值
df['cos_wind_direction'] = np.cos(np.deg2rad(df['風向']))            # 角度轉為cos值

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

# 提取座標欄位 (假設這 12 個欄位都是字串形式 "(經度, 緯度)")
target_columns = [col for col in df.columns if '(' in col and ')' in col][:12]
print("所有座標點：", target_columns)

# 替換 DataFrame 欄位名稱為英文（供 LightGBM 使用）
df_tree = df.rename(columns=feature_mapping)

# 設定 X (特徵) 與 y (目標)
X = df_original[list(feature_mapping.keys())]
y = df_original[target_columns]  # 這裡也可用 df[target_columns]，視需求而定

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tree = X_train.rename(columns=feature_mapping)
X_test_tree = X_test.rename(columns=feature_mapping)

# 定義類別型特徵
cat_features = ['Holiday', 'Weekday', 'Month', 'Hour']

# 解析字串形式 "(經度, 緯度)" 的小函式
def parse_coord_string(coord_str):
    """
    將單個字串座標，如"(121.5, 25.0)"，解析成 (lon, lat) (float, float)
    """
    coord_str = coord_str.strip("() ")
    lon_str, lat_str = coord_str.split(",")
    return float(lon_str), float(lat_str)

# 先取得決策樹使用的特徵順序（依據 X_train_tree 的欄位順序）
features_order = list(X_train_tree.columns)

# 定義一個函數，利用遞迴遍歷決策樹，累積各特徵的 split_gain
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

# 用來儲存每個「網格/目標」的最佳決策樹表示向量、代表經緯度、網格ID
grid_ids = []
tree_vectors = []
geo_coords = []
predictions = {}

# 逐一為每個座標欄位 (target) 訓練模型
for target in target_columns:
    print(f"訓練與預測 {target} 的決策樹...")

    # 1) 建立 LightGBM Dataset
    train_data = lgb.Dataset(X_train_tree, label=y_train[target], categorical_feature=cat_features)
    test_data = lgb.Dataset(X_test_tree, label=y_test[target], reference=train_data, categorical_feature=cat_features)

    # 2) 設定參數
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'seed': 42
    }
    evals_result = {}

    # 3) 訓練模型
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        valid_names=["valid_0"],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=True),
                   lgb.record_evaluation(evals_result),
                   lgb.log_evaluation(25)]
    )

    # 4) 預測
    y_pred = lgb_model.predict(X_test_tree, num_iteration=lgb_model.best_iteration)
    predictions[target] = y_pred
    print(f"LightGBM 總樹數: {lgb_model.num_trees()}")

    # 5) 繪製學習曲線
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

    # 6) 找出最佳決策樹 (以總 split_gain 最大的那棵)
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

    # 繪製最後決策樹 *這段註解不要刪掉*
    # plt.figure(figsize=(20, 10))
    # lgb.plot_tree(lgb_model, tree_index=lgb_model.best_iteration-1, show_info=['split_gain'])
    # plt.title(f"Best Decision Tree for {target}")
    # tree_plot_path = os.path.join(result_dir, f"best_tree_{target.replace(',', '_').replace(' ', '')}.png")
    # plt.savefig(tree_plot_path, dpi=900, bbox_inches="tight")
    # plt.close()
    # print(f"最佳決策樹圖已儲存至: {tree_plot_path}")

    # 7) 提取該最佳樹的特徵增益向量
    vector = np.zeros(len(features_order))
    best_tree = tree_info[best_tree_index]["tree_structure"]
    vector = get_feature_gain_vector(best_tree, vector)
    tree_vectors.append(vector)

    # 8) 直接取該 target 欄位的代表經緯度（因為每個 target 就包含經緯度）
    geo_coords.append(target)
    grid_ids.append(target)

    # 9) SHAP 分析
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

# ------------------------
# 分群：利用各網格最佳決策樹的特徵增益向量進行 KMeans 分群，並選出 k 個代表
k = 3
tree_vectors = np.array(tree_vectors)
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(tree_vectors)
centroids = kmeans.cluster_centers_

# 選出每群中距離質心最近的網格作為代表
rep_indices = []
for i in range(k):
    cluster_indices = np.where(cluster_labels == i)[0]
    if len(cluster_indices) > 0:
        dists = np.linalg.norm(tree_vectors[cluster_indices] - centroids[i], axis=1)
        rep_idx = cluster_indices[np.argmin(dists)]
        rep_indices.append(rep_idx)
    else:
        rep_indices.append(None)

rep_geo_coords = [geo_coords[idx] for idx in rep_indices if idx is not None]
print("代表經緯度：")
for i, coord in enumerate(rep_geo_coords):
    print(f"群 {i}: {coord}")
# TODO: 視覺化k-means分群樣態，我想知道他怎麼分群的


# 繪製分群結果的地理散佈圖
# 將 geo_coords（字串列表）轉為數值型 tuple，假設格式為 "(lon, lat)"
# 先建構解析後的 (lon, lat)
parsed_coords = [tuple(map(float, coord.strip("()").split(","))) for coord in geo_coords]
all_lons = [coord[0] for coord in parsed_coords]  # 經度
all_lats = [coord[1] for coord in parsed_coords]  # 緯度

# 建立一個顏色規範 (norm)，使 colorbar 從 0 ~ (k-1)
norm = mpl.colors.Normalize(vmin=0, vmax=k-1)

plt.figure(figsize=(10, 8))

# 繪製所有網格點散佈圖
#   c=cluster_labels 會根據群集標籤來上色
sc = plt.scatter(
    all_lons, all_lats,
    c=cluster_labels,
    cmap='viridis',
    norm=norm,      # 套用顏色規範
    s=50,
    alpha=0.7
)

# 繪製代表網格 (X)
#   取出代表網格的座標 & 它們對應的群集標籤
rep_coords = [parsed_coords[idx] for idx in rep_indices if idx is not None]
rep_labels = [cluster_labels[idx] for idx in rep_indices if idx is not None]
rep_lons = [coord[0] for coord in rep_coords]
rep_lats = [coord[1] for coord in rep_coords]

plt.scatter(
    rep_lons, rep_lats,
    c=rep_labels,
    cmap='viridis',
    norm=norm,
    marker='X',
    s=200,
    edgecolor='black',  # 用黑邊讓代表點更明顯
    alpha=1.0
)

# 關閉科學記號，避免經度顯示為 1.215e+02
plt.ticklabel_format(useOffset=False, style='plain', axis='both')

# 餘標籤 & 色卡設定
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Clustering of Grids Based on Decision Tree Features")

# 使用與散佈圖 sc 相同的 colormap/norm 來繪製 colorbar
plt.colorbar(sc, label="Cluster Label")

clustering_path = os.path.join(result_dir, "geo_clustering.png")
plt.savefig(clustering_path, dpi=900, bbox_inches="tight")
plt.close()
print(f"地理分群圖已儲存至: {clustering_path}")



# 繪製分群結果與特徵關係圖
# 建立英文→中文的反向對照表
reverse_mapping = {v: k for k, v in feature_mapping.items()}

top_feature = 12

# 假設您只想顯示 features_order 裡的特徵
top_cols = features_order[:top_feature]

# 將 tree_vectors 裡的前幾維做成 DataFrame
df_features = pd.DataFrame(
    tree_vectors[:, :top_feature],  # 只取前幾維
    index=grid_ids,
    columns=top_cols
)

# 加入分群標籤（字串格式）
df_features['Cluster'] = cluster_labels.astype(str)

# 將欄位名稱從英文改回中文
df_features.rename(columns=lambda c: reverse_mapping.get(c, c), inplace=True)

# 繪製平行座標圖
# 右上角有標0,1,2,...改成標代表座標
plt.figure(figsize=(12, 8))
parallel_coordinates(df_features, class_column='Cluster', colormap='viridis')
plt.title("Parallel Coordinates Plot by Cluster")
plt.tight_layout()
feature_cluster_path = os.path.join(result_dir, "feature_cluster_parallel_coordinates.png")
plt.savefig(feature_cluster_path, dpi=900, bbox_inches="tight")
plt.close()
print(f"特徵與分群關係的平行座標圖已儲存至: {feature_cluster_path}")

# TODO: 列出作為代表的K個座標存成txt
# TODO: 存下這K個決策樹模型



'''
    蚁群算法ACO
    适用：
'''
'''
    在随机森林代码的基础上加入了ACO的特征选择环节
'''
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import preprocessing, metrics
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import plot_tree
# import warnings
#
# warnings.filterwarnings("ignore")
#
# # ---------------- 1. 读取数据 ----------------
# file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
# dataset = pd.read_csv(file_name)
#
# # ---- 基础价差类特征 ----
# dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']
# dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']
# dataset['Return'] = dataset['Close'].pct_change()
# dataset['Volume_Change'] = dataset['Volume'].pct_change()
#
# # ---- 移动平均和动量类特征 ----
# dataset['MA_5'] = dataset['Close'].rolling(window=5).mean()
# dataset['MA_10'] = dataset['Close'].rolling(window=10).mean()
# dataset['MA_20'] = dataset['Close'].rolling(window=20).mean()
#
# dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
# dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)
#
# # ---- 波动性特征 ----
# dataset['Rolling_STD_5'] = dataset['Close'].rolling(window=5).std()
# dataset['Rolling_STD_10'] = dataset['Close'].rolling(window=10).std()
#
# # ---- 移动平均收敛差指标（MACD） ----
# ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
# ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
# dataset['MACD'] = ema12 - ema26
#
# # ---- 相对强弱指数（RSI） ----
# delta = dataset['Close'].diff()
# gain = np.where(delta > 0, delta, 0)
# loss = np.where(delta < 0, -delta, 0)
# avg_gain = pd.Series(gain).rolling(window=14).mean()
# avg_loss = pd.Series(loss).rolling(window=14).mean()
# rs = avg_gain / avg_loss
# dataset['RSI_14'] = 100 - (100 / (1 + rs))
#
# dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')
#
# dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
# dataset.dropna(inplace=True)
# dataset.reset_index(drop=True, inplace=True)
#
# features = [
#     'Open', 'High', 'Low', 'Close', 'Volume',
#     'High_Low_Spread', 'Close_Open_Change', 'Return', 'Volume_Change',
#     'MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'Momentum_10',
#     'Rolling_STD_5', 'Rolling_STD_10', 'MACD', 'RSI_14'
# ]
# X = dataset[features].values
# y = dataset['Up_Down']
#
# # ---------------- 6. 划分训练集与测试集 ----------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0
# )
#
# # ---------------- 7. ACO 特征选择 ----------------
# num_ants = 20
# num_iterations = 30
# evaporation_rate = 0.2
# alpha = 1.0
# num_features = X_train.shape[1]
# pheromone = np.ones(num_features)
#
# best_subset = None
# best_score = 0
#
# for it in range(num_iterations):
#     ant_solutions = []
#     ant_scores = []
#     for ant in range(num_ants):
#         probs = pheromone ** alpha
#         probs = probs / probs.sum()
#         subset = np.random.rand(num_features) < probs
#         if subset.sum() == 0:
#             subset[np.random.randint(0, num_features)] = True
#
#         X_sub = X_train[:, subset]
#         rf = RandomForestClassifier(n_estimators=100, random_state=0)
#         score = cross_val_score(rf, X_sub, y_train, cv=3).mean()
#
#         ant_solutions.append(subset)
#         ant_scores.append(score)
#
#         if score > best_score:
#             best_score = score
#             best_subset = subset
#
#     pheromone = (1 - evaporation_rate) * pheromone
#     for subset, score in zip(ant_solutions, ant_scores):
#         pheromone[subset] += score
#
#     print(f"Iteration {it + 1}/{num_iterations}  Best score so far: {best_score:.4f}")
#
# # ---------------- 8. 用 ACO 筛选特征训练最终随机森林 ----------------
# selected_features = np.where(best_subset)[0]
# selected_feature_names = [features[i] for i in selected_features]
#
# print("Selected features:", selected_feature_names)
# print("Number of selected features:", len(selected_features))
#
# X_train_sel = X_train[:, selected_features]
# X_test_sel = X_test[:, selected_features]
#
# rf_final = RandomForestClassifier(n_estimators=100, random_state=0)
# rf_final.fit(X_train_sel, y_train)
#
# y_pred = rf_final.predict(X_test_sel)
# acc = metrics.accuracy_score(y_test, y_pred)
# print("Random Forest Accuracy (after ACO feature selection):", acc)
#
# # ---------------- 9. 可视化：混淆矩阵 ----------------
# cm = metrics.confusion_matrix(y_test, y_pred, labels=rf_final.classes_)
#
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=rf_final.classes_, yticklabels=rf_final.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
#
# # ---------------- 10. 可视化：特征重要性 ----------------
# importances = rf_final.feature_importances_
#
# plt.figure(figsize=(5, 4))
# sns.barplot(x=importances, y=selected_feature_names)
# plt.title('Feature Importances')
# plt.show()
#
# # ---------------- 11. 可视化：单棵树结构（可选） ----------------
# plt.figure(figsize=(15, 10))
# plot_tree(rf_final.estimators_[0], feature_names=selected_feature_names,
#           class_names=rf_final.classes_, filled=True, max_depth=4)
# plt.title('Example Tree from Random Forest')
# plt.show()

'''
    速通版
    
结果：
Iteration 1/10  Best score so far: 0.7228
Iteration 2/10  Best score so far: 0.7391
Iteration 3/10  Best score so far: 0.7391
Iteration 4/10  Best score so far: 0.7391
Iteration 5/10  Best score so far: 0.7609
Iteration 6/10  Best score so far: 0.7609
Iteration 7/10  Best score so far: 0.7717
Iteration 8/10  Best score so far: 0.7717
Iteration 9/10  Best score so far: 0.7717
Iteration 10/10  Best score so far: 0.7717
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import warnings

warnings.filterwarnings("ignore")

# ---------------- 1. 读取数据 ----------------
file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
dataset = pd.read_csv(file_name)

# ---- 基础价差类特征 ----
dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']
dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']
dataset['Return'] = dataset['Close'].pct_change()
dataset['Volume_Change'] = dataset['Volume'].pct_change()

# ---- 移动平均和动量类特征 ----
dataset['MA_5'] = dataset['Close'].rolling(window=5).mean()
dataset['MA_10'] = dataset['Close'].rolling(window=10).mean()
dataset['MA_20'] = dataset['Close'].rolling(window=20).mean()
dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)

# ---- 波动性特征 ----
dataset['Rolling_STD_5'] = dataset['Close'].rolling(window=5).std()
dataset['Rolling_STD_10'] = dataset['Close'].rolling(window=10).std()

# ---- MACD 和 RSI ----
ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
dataset['MACD'] = ema12 - ema26

delta = dataset['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
dataset['RSI_14'] = 100 - (100 / (1 + rs))

dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')

dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)

features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'High_Low_Spread', 'Close_Open_Change', 'Return', 'Volume_Change',
    'MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'Momentum_10',
    'Rolling_STD_5', 'Rolling_STD_10', 'MACD', 'RSI_14'
]
X = dataset[features].values
y = dataset['Up_Down']

# ---------------- 6. 划分训练集、验证集和测试集 ----------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=0
)

# ---------------- 7. ACO 特征选择（快速模式） ----------------
num_ants = 10
num_iterations = 10
evaporation_rate = 0.2
alpha = 1.0
num_features = X_train.shape[1]
pheromone = np.ones(num_features)

best_subset = None
best_score = 0

for it in range(num_iterations):
    ant_solutions = []
    ant_scores = []
    for ant in range(num_ants):
        # 限制每只蚂蚁随机选 5~10 个特征
        subset = np.zeros(num_features, dtype=bool)
        selected_indices = np.random.choice(num_features, size=np.random.randint(5, 11), replace=False)
        subset[selected_indices] = True

        X_sub = X_train[:, subset]
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        rf.fit(X_sub, y_train)
        score = rf.score(X_val[:, subset], y_val)

        ant_solutions.append(subset)
        ant_scores.append(score)

        if score > best_score:
            best_score = score
            best_subset = subset

    pheromone = (1 - evaporation_rate) * pheromone
    for subset, score in zip(ant_solutions, ant_scores):
        pheromone[subset] += score

    print(f"Iteration {it + 1}/{num_iterations}  Best score so far: {best_score:.4f}")

'''
    粒子群优化PSO
    适用：
'''

'''
优化的是 随机森林的两个超参数：
    # n_estimators（树的数量）
    # max_depth（最大深度）
'''

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from pyswarm import pso  # pip install pyswarm
# from sklearn.preprocessing import LabelEncoder
# import warnings
# warnings.filterwarnings("ignore")
#
# # ---------------- 1. 读取数据 ----------------
# file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
# dataset = pd.read_csv(file_name)
#
# # ---- 特征工程 ----
# dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']
# dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']
# dataset['Return'] = dataset['Close'].pct_change()
# dataset['Volume_Change'] = dataset['Volume'].pct_change()
# dataset['MA_5'] = dataset['Close'].rolling(5).mean()
# dataset['MA_10'] = dataset['Close'].rolling(10).mean()
# dataset['MA_20'] = dataset['Close'].rolling(20).mean()
# dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
# dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)
# dataset['Rolling_STD_5'] = dataset['Close'].rolling(5).std()
# dataset['Rolling_STD_10'] = dataset['Close'].rolling(10).std()
# ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
# ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
# dataset['MACD'] = ema12 - ema26
# delta = dataset['Close'].diff()
# gain = np.where(delta > 0, delta, 0)
# loss = np.where(delta < 0, -delta, 0)
# avg_gain = pd.Series(gain).rolling(window=14).mean()
# avg_loss = pd.Series(loss).rolling(window=14).mean()
# rs = avg_gain / avg_loss
# dataset['RSI_14'] = 100 - (100 / (1 + rs))
# dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')
#
# dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
# dataset.dropna(inplace=True)
# dataset.reset_index(drop=True, inplace=True)
#
# features = [
#     'Open','High','Low','Close','Volume',
#     'High_Low_Spread','Close_Open_Change','Return','Volume_Change',
#     'MA_5','MA_10','MA_20','Momentum_5','Momentum_10',
#     'Rolling_STD_5','Rolling_STD_10','MACD','RSI_14'
# ]
# X = dataset[features].values
# y = LabelEncoder().fit_transform(dataset['Up_Down'])  # 转为 0/1
#
# # ---------------- 2. 划分训练集与测试集 ----------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0
# )
#
# # ---------------- 3. 定义适应度函数 ----------------
# def fitness(params):
#     # params = [n_estimators, max_depth]
#     n_estimators = int(params[0])
#     max_depth = int(params[1])
#     if max_depth <= 0:  # 防止出现负值
#         max_depth = None
#
#     clf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         random_state=0,
#         n_jobs=-1
#     )
#     # 使用 3 折交叉验证
#     score = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()
#     return -score  # PSO 是最小化问题，所以取负号
#
# # ---------------- 4. 粒子群优化 ----------------
# lb = [10, 1]    # 下界 [n_estimators_min, max_depth_min]
# ub = [200, 20]  # 上界 [n_estimators_max, max_depth_max]
#
# best_params, best_score = pso(fitness, lb, ub, swarmsize=20, maxiter=30)
#
# best_n_estimators = int(best_params[0])
# best_max_depth = int(best_params[1])
# print(f"优化后的参数：n_estimators={best_n_estimators}, max_depth={best_max_depth}")
# print(f"交叉验证最佳准确率：{-best_score:.4f}")
#
# # ---------------- 5. 使用最优参数训练随机森林 ----------------
# rf_best = RandomForestClassifier(
#     n_estimators=best_n_estimators,
#     max_depth=best_max_depth,
#     random_state=0,
#     n_jobs=-1
# )
# rf_best.fit(X_train, y_train)
#
# # ---------------- 6. 测试集评估 ----------------
# from sklearn import metrics
# y_pred = rf_best.predict(X_test)
# acc = metrics.accuracy_score(y_test, y_pred)
# print("测试集准确率:", acc)

'''
用 粒子群优化 (PSO) 来做 特征选择，找到对预测涨跌最有用的特征组合
'''
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from pyswarm import pso  # pip install pyswarm
# from sklearn.preprocessing import LabelEncoder
# import warnings
# warnings.filterwarnings("ignore")
#
# # ---------------- 1. 读取数据 ----------------
# file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
# dataset = pd.read_csv(file_name)
#
# # ---- 特征工程 ----
# dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']
# dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']
# dataset['Return'] = dataset['Close'].pct_change()
# dataset['Volume_Change'] = dataset['Volume'].pct_change()
# dataset['MA_5'] = dataset['Close'].rolling(5).mean()
# dataset['MA_10'] = dataset['Close'].rolling(10).mean()
# dataset['MA_20'] = dataset['Close'].rolling(20).mean()
# dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
# dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)
# dataset['Rolling_STD_5'] = dataset['Close'].rolling(5).std()
# dataset['Rolling_STD_10'] = dataset['Close'].rolling(10).std()
# ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
# ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
# dataset['MACD'] = ema12 - ema26
# delta = dataset['Close'].diff()
# gain = np.where(delta > 0, delta, 0)
# loss = np.where(delta < 0, -delta, 0)
# avg_gain = pd.Series(gain).rolling(window=14).mean()
# avg_loss = pd.Series(loss).rolling(window=14).mean()
# rs = avg_gain / avg_loss
# dataset['RSI_14'] = 100 - (100 / (1 + rs))
# dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')
#
# dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
# dataset.dropna(inplace=True)
# dataset.reset_index(drop=True, inplace=True)
#
# features = [
#     'Open','High','Low','Close','Volume',
#     'High_Low_Spread','Close_Open_Change','Return','Volume_Change',
#     'MA_5','MA_10','MA_20','Momentum_5','Momentum_10',
#     'Rolling_STD_5','Rolling_STD_10','MACD','RSI_14'
# ]
# X = dataset[features].values
# y = LabelEncoder().fit_transform(dataset['Up_Down'])
#
# # ---------------- 2. 划分训练集与测试集 ----------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0
# )
#
# # ---------------- 3. 定义适应度函数（特征选择） ----------------
# def fitness(feature_mask):
#     # feature_mask 是粒子的位置，每个值 [0,1]
#     # 将大于0.5的视为选择该特征
#     mask = feature_mask > 0.5
#     if np.sum(mask) == 0:  # 防止没有特征被选中
#         return 1.0
#     X_sub = X_train[:, mask]
#     clf = RandomForestClassifier(n_estimators=100, random_state=0)
#     score = cross_val_score(clf, X_sub, y_train, cv=3, scoring='accuracy').mean()
#     return -score  # 最小化问题
#
# # ---------------- 4. 粒子群优化 ----------------
# lb = [0] * len(features)  # 每个特征最小值 0
# ub = [1] * len(features)  # 每个特征最大值 1
#
# best_mask, best_score = pso(fitness, lb, ub, swarmsize=30, maxiter=40)
#
# # ---------------- 5. 输出结果 ----------------
# selected_features = [f for f, m in zip(features, best_mask > 0.5) if m]
# print("PSO选择的特征：", selected_features)
# print("交叉验证最佳准确率：", -best_score)
#
# # ---------------- 6. 使用选出的特征训练随机森林 ----------------
# X_train_sub = X_train[:, best_mask > 0.5]
# X_test_sub = X_test[:, best_mask > 0.5]
#
# rf_best = RandomForestClassifier(n_estimators=100, random_state=0)
# rf_best.fit(X_train_sub, y_train)
#
# from sklearn import metrics
# y_pred = rf_best.predict(X_test_sub)
# acc = metrics.accuracy_score(y_test, y_pred)
# print("测试集准确率:", acc)

'''
速通版，上面那个太慢了

结果：
Stopping search: maximum iterations reached --> 10
PSO选择的特征： ['Open', 'Low', 'Close_Open_Change', 'Return', 'MA_5', 'Momentum_5', 'Rolling_STD_10', 'RSI_14']
交叉验证最佳准确率： 0.7478260869565218
测试集准确率: 0.70995670995671

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from pyswarm import pso  # pip install pyswarm
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ---------------- 1. 读取数据 ----------------
file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
dataset = pd.read_csv(file_name)

# ---- 特征工程 ----
dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']
dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']
dataset['Return'] = dataset['Close'].pct_change()
dataset['Volume_Change'] = dataset['Volume'].pct_change()
dataset['MA_5'] = dataset['Close'].rolling(5).mean()
dataset['MA_10'] = dataset['Close'].rolling(10).mean()
dataset['MA_20'] = dataset['Close'].rolling(20).mean()
dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)
dataset['Rolling_STD_5'] = dataset['Close'].rolling(5).std()
dataset['Rolling_STD_10'] = dataset['Close'].rolling(10).std()
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
    'Open','High','Low','Close','Volume',
    'High_Low_Spread','Close_Open_Change','Return','Volume_Change',
    'MA_5','MA_10','MA_20','Momentum_5','Momentum_10',
    'Rolling_STD_5','Rolling_STD_10','MACD','RSI_14'
]
X = dataset[features].values
y = LabelEncoder().fit_transform(dataset['Up_Down'])

# ---------------- 2. 划分训练集与测试集 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------------- 3. 定义适应度函数（特征选择） ----------------
def fitness(feature_mask):
    mask = feature_mask > 0.5
    if np.sum(mask) == 0:
        return 1.0
    X_sub = X_train[:, mask]
    clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    # 快速评估：2 折交叉验证
    score = cross_val_score(clf, X_sub, y_train, cv=2, scoring='accuracy').mean()
    return -score

# ---------------- 4. 粒子群优化（快速版） ----------------
lb = [0] * len(features)
ub = [1] * len(features)

best_mask, best_score = pso(fitness, lb, ub, swarmsize=10, maxiter=10)

# ---------------- 5. 输出结果 ----------------
selected_features = [f for f, m in zip(features, best_mask > 0.5) if m]
print("PSO选择的特征：", selected_features)
print("交叉验证最佳准确率：", -best_score)

# ---------------- 6. 使用选出的特征训练随机森林 ----------------
X_train_sub = X_train[:, best_mask > 0.5]
X_test_sub = X_test[:, best_mask > 0.5]

rf_best = RandomForestClassifier(n_estimators=50, random_state=0)
rf_best.fit(X_train_sub, y_train)

from sklearn import metrics
y_pred = rf_best.predict(X_test_sub)
acc = metrics.accuracy_score(y_test, y_pred)
print("测试集准确率:", acc)



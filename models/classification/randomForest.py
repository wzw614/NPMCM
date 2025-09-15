# # 🌳 Random Forest Classifier
# #
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import preprocessing, metrics
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import plot_tree
# import warnings
# warnings.filterwarnings("ignore")
#
# # ---------------- 1. 读取数据 ----------------
# file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
# dataset = pd.read_csv(file_name)
#
# # ---------------- 2. 构造标签列 ----------------
# dataset['Return'] = dataset['Close'].pct_change()
# dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')
#
# # ---------------- 3. 构造特征列 ----------------
# dataset['Open_N']   = np.where(dataset['Open'].shift(-1)   > dataset['Open'],   'Up', 'Down')
# dataset['Volume_N'] = np.where(dataset['Volume'].shift(-1) > dataset['Volume'], 'Positive', 'Negative')
#
# # 去掉缺失值（因为有 shift 操作）
# dataset = dataset.dropna()
#
# # ---------------- 4. 划分特征X、目标y ----------------
# X = dataset[['Open', 'Open_N', 'Volume_N']].values
# y = dataset['Up_Down']
#
# # ---------------- 5. 标签编码 ----------------
# le_Open = preprocessing.LabelEncoder()
# le_Open.fit(['Up','Down'])
# X[:,1] = le_Open.transform(X[:,1])
#
# le_Volume = preprocessing.LabelEncoder()
# le_Volume.fit(['Positive','Negative'])
# X[:,2] = le_Volume.transform(X[:,2])
#
# # ---------------- 6. 划分训练集与测试集 ----------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0
# )
#
# # ---------------- 7. 训练随机森林模型 ----------------
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=None,
#     random_state=0
# )
# rf.fit(X_train, y_train)
#
# # ---------------- 8. 预测与评估 ----------------
# y_pred = rf.predict(X_test)
# acc = metrics.accuracy_score(y_test, y_pred)
# print("Random Forest Accuracy:", acc)
#
# # ---------------- 9. 可视化：混淆矩阵 ----------------
# cm = metrics.confusion_matrix(y_test, y_pred, labels=rf.classes_)
#
# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=rf.classes_, yticklabels=rf.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
#
# # ---------------- 10. 可视化：特征重要性 ----------------
# feature_names = ['Open', 'Open_N', 'Volume_N']
# importances = rf.feature_importances_
#
# plt.figure(figsize=(5,4))
# sns.barplot(x=importances, y=feature_names)
# plt.title('Feature Importances')
# plt.show()
#
# # ---------------- 11. 可视化：单棵树结构（可选） ----------------
# plt.figure(figsize=(15, 10))
# plot_tree(rf.estimators_[0], feature_names=feature_names,
#           class_names=rf.classes_, filled=True, max_depth=4)
# plt.title('Example Tree from Random Forest')
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# ---------------- 1. 读取数据 ----------------
file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
dataset = pd.read_csv(file_name)

# ---- 基础价差类特征 ----
dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']                # 当日最高-最低
dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']            # 收盘-开盘
dataset['Return'] = dataset['Close'].pct_change()                             # 日收益率
dataset['Volume_Change'] = dataset['Volume'].pct_change()                     # 成交量变化率

# ---- 移动平均和动量类特征 ----
dataset['MA_5'] = dataset['Close'].rolling(window=5).mean()
dataset['MA_10'] = dataset['Close'].rolling(window=10).mean()
dataset['MA_20'] = dataset['Close'].rolling(window=20).mean()

dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)

# ---- 波动性特征 ----
dataset['Rolling_STD_5'] = dataset['Close'].rolling(window=5).std()
dataset['Rolling_STD_10'] = dataset['Close'].rolling(window=10).std()

# ---- 移动平均收敛差指标（MACD） ----
ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
dataset['MACD'] = ema12 - ema26

# ---- 相对强弱指数（RSI） ----
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
y = dataset['Up_Down']

# ---------------- 6. 划分训练集与测试集 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
# ---------------- 7. 训练随机森林模型 ----------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=0
)
rf.fit(X_train, y_train)
# ---------------- 8. 预测与评估 ----------------
y_pred = rf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", acc)

# ---------------- 9. 可视化：混淆矩阵 ----------------
cm = metrics.confusion_matrix(y_test, y_pred, labels=rf.classes_)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ---------------- 10. 可视化：特征重要性 ----------------
feature_names = features
importances = rf.feature_importances_

plt.figure(figsize=(5,4))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.show()

# ---------------- 11. 可视化：单棵树结构（可选） ----------------
plt.figure(figsize=(15, 10))
plot_tree(rf.estimators_[0], feature_names=feature_names,
          class_names=rf.classes_, filled=True, max_depth=4)
plt.title('Example Tree from Random Forest')
plt.show()




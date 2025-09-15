#%% md
# XGBoost 分类完整流程

# ---------------- 0. 导入库 ----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

# ---------------- 1. 读取数据 ----------------
file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
dataset = pd.read_csv(file_name)

# ---------------- 2. 构造特征 ----------------
# 基础价差类
dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']
dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']
dataset['Return'] = dataset['Close'].pct_change()
dataset['Volume_Change'] = dataset['Volume'].pct_change()

# 移动平均与动量
dataset['MA_5'] = dataset['Close'].rolling(5).mean()
dataset['MA_10'] = dataset['Close'].rolling(10).mean()
dataset['MA_20'] = dataset['Close'].rolling(20).mean()
dataset['Momentum_5'] = dataset['Close'] - dataset['Close'].shift(5)
dataset['Momentum_10'] = dataset['Close'] - dataset['Close'].shift(10)

# 波动性特征
dataset['Rolling_STD_5'] = dataset['Close'].rolling(5).std()
dataset['Rolling_STD_10'] = dataset['Close'].rolling(10).std()

# MACD
ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
dataset['MACD'] = ema12 - ema26

# RSI
delta = dataset['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14).mean()
avg_loss = pd.Series(loss).rolling(14).mean()
rs = avg_gain / avg_loss
dataset['RSI_14'] = 100 - (100 / (1 + rs))

# 目标变量
dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')

# ---------------- 3. 清洗数据 ----------------
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)

# ---------------- 4. 特征和目标 ----------------
features = [
    'Open','High','Low','Close','Volume',
    'High_Low_Spread','Close_Open_Change','Return','Volume_Change',
    'MA_5','MA_10','MA_20','Momentum_5','Momentum_10',
    'Rolling_STD_5','Rolling_STD_10','MACD','RSI_14'
]

X = dataset[features].values
y = dataset['Up_Down']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Down'->0, 'Up'->1


# ---------------- 5. 划分训练集与测试集 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------- 6. 初始化 XGBoost ----------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# ---------------- 7. 模型训练 ----------------
xgb_model.fit(X_train, y_train)

# ---------------- 8. 预测与评估 ----------------
y_pred = xgb_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)

# 混淆矩阵
cm = metrics.confusion_matrix(y_test, y_pred, labels=xgb_model.classes_)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=xgb_model.classes_, yticklabels=xgb_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ---------------- 9. 特征重要性 ----------------
importances = xgb_model.feature_importances_
plt.figure(figsize=(6,6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.show()

# ---------------- 10. 可选：网格搜索调参 ----------------
param_grid = {
    'max_depth': [3,4,5],
    'n_estimators': [100,200,300],
    'learning_rate': [0.05,0.1,0.2]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='binary:logistic',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# 用最佳参数预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Test Accuracy with Best Model:", metrics.accuracy_score(y_test, y_pred_best))

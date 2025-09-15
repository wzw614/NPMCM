# '''
#     遗传算法GA
# '''
#
# # Genetic Algorithm for Hyperparameter Optimization
# # 使用遗传算法查找 随机森林的超参数
# 从头实现
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import preprocessing, metrics
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import plot_tree
# from sklearn.preprocessing import LabelEncoder
# import warnings
# import random
# warnings.filterwarnings("ignore")
#
#
# # ---------------- 1. 读取数据 ----------------
# file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
# dataset = pd.read_csv(file_name)
#
# # ---- 基础价差类特征 ----
# dataset['High_Low_Spread'] = dataset['High'] - dataset['Low']                # 当日最高-最低
# dataset['Close_Open_Change'] = dataset['Close'] - dataset['Open']            # 收盘-开盘
# dataset['Return'] = dataset['Close'].pct_change()                             # 日收益率
# dataset['Volume_Change'] = dataset['Volume'].pct_change()                     # 成交量变化率
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
# # ========== 数据准备 ==========
# features = [
#     'Open','High','Low','Close','Volume',
#     'High_Low_Spread','Close_Open_Change','Return','Volume_Change',
#     'MA_5','MA_10','MA_20','Momentum_5','Momentum_10',
#     'Rolling_STD_5','Rolling_STD_10','MACD','RSI_14'
# ]
# X = dataset[features].values
# y = dataset['Up_Down'].map({'Up':1,'Down':0}).values
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # ========== 遗传算法配置 ==========
# POP_SIZE = 10        # 种群数量
# N_GEN = 10            # 迭代代数
# MUT_RATE = 0.2        # 变异率
# CX_RATE = 0.5         # 交叉率
#
# # 超参数搜索空间
# n_estimators_range = (50, 300)
# max_depth_range = (2, 10)
# min_samples_split_range = (2, 10)
#
# # ========== 适应度函数 ==========
# def fitness(individual):
#     n_estimators, max_depth, min_samples_split = individual
#     model = RandomForestClassifier(
#         n_estimators=int(n_estimators),
#         max_depth=int(max_depth),
#         min_samples_split=int(min_samples_split),
#         random_state=42
#     )
#     scores = cross_val_score(model, X_train, y_train, cv=3)
#     return scores.mean()
#
# # ========== 初始化种群 ==========
# def init_population():
#     return [
#         [
#             random.randint(*n_estimators_range),
#             random.randint(*max_depth_range),
#             random.randint(*min_samples_split_range)
#         ]
#         for _ in range(POP_SIZE)
#     ]
#
# # ========== 选择（锦标赛）==========
# def select(pop, fits):
#     # 随机挑选两个，取适应度高的
#     i, j = random.sample(range(len(pop)), 2)
#     return pop[i] if fits[i] > fits[j] else pop[j]
#
# # ========== 交叉（单点）==========
# def crossover(parent1, parent2):
#     if random.random() < CX_RATE:
#         point = random.randint(1, len(parent1)-1)
#         child1 = parent1[:point] + parent2[point:]
#         child2 = parent2[:point] + parent1[point:]
#         return child1, child2
#     return parent1[:], parent2[:]
#
# # ========== 变异 ==========
# def mutate(individual):
#     if random.random() < MUT_RATE:
#         idx = random.randint(0, len(individual)-1)
#         if idx == 0:
#             individual[idx] = random.randint(*n_estimators_range)
#         elif idx == 1:
#             individual[idx] = random.randint(*max_depth_range)
#         else:
#             individual[idx] = random.randint(*min_samples_split_range)
#
# # ========== 主进化循环 ==========
# population = init_population()
# for gen in range(N_GEN):
#     fitnesses = [fitness(ind) for ind in population]
#     new_population = []
#     while len(new_population) < POP_SIZE:
#         p1 = select(population, fitnesses)
#         p2 = select(population, fitnesses)
#         c1, c2 = crossover(p1, p2)
#         mutate(c1)
#         mutate(c2)
#         new_population.extend([c1, c2])
#     population = new_population[:POP_SIZE]
#     best_fit = max(fitnesses)
#     print(f"Generation {gen+1} | Best fitness: {best_fit:.4f}")
#
# # ========== 在测试集上评估最佳个体 ==========
# best_idx = np.argmax([fitness(ind) for ind in population])
# best_ind = population[best_idx]
# print("Best hyperparameters:", best_ind)
#
# best_model = RandomForestClassifier(
#     n_estimators=int(best_ind[0]),
#     max_depth=int(best_ind[1]),
#     min_samples_split=int(best_ind[2]),
#     random_state=42
# )
# best_model.fit(X_train, y_train)
# test_acc = best_model.score(X_test, y_test)
# print("Test Accuracy:", test_acc)


# 使用PyGAD包——找超参数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pygad
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

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

# ---- MACD ----
ema12 = dataset['Close'].ewm(span=12, adjust=False).mean()
ema26 = dataset['Close'].ewm(span=26, adjust=False).mean()
dataset['MACD'] = ema12 - ema26

# ---- RSI ----
delta = dataset['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
dataset['RSI_14'] = 100 - (100 / (1 + rs))

# ---- 标签 ----
dataset['Up_Down'] = np.where(dataset['Return'].shift(-1) > dataset['Return'], 'Up', 'Down')

# ---- 清理 ----
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

# ---------------- 2. 划分训练集与测试集 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------------- 3. 定义GA适应度函数 ----------------
def fitness_func(ga_instance, solution, solution_idx):
    # 将解（solution）解包为参数，例如 n_estimators 和 max_depth
    n_estimators = int(solution[0])
    max_depth = int(solution[1])

    # 训练模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算准确率作为适应度
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc

# ---------------- 4. 配置并运行PyGAD ----------------
gene_space = [
    {'low': 50, 'high': 300},    # n_estimators
    {'low': 2, 'high': 20},      # max_depth
    {'low': 2, 'high': 10},      # min_samples_split
    {'low': 1, 'high': 5}        # min_samples_leaf
]

ga = pygad.GA(
    num_generations=20,
    num_parents_mating=6,
    fitness_func=fitness_func,
    sol_per_pop=12,
    num_genes=4,
    gene_space=gene_space,
    mutation_percent_genes=25
)

print("开始遗传搜索超参数...")
ga.run()

# ---------------- 5. 使用最优参数训练最终模型 ----------------
best_solution, best_fitness, _ = ga.best_solution()
best_n_estimators = int(best_solution[0])
best_max_depth = int(best_solution[1])
best_min_samples_split = int(best_solution[2])
best_min_samples_leaf = int(best_solution[3])

print("最优参数：")
print(f"n_estimators={best_n_estimators}, max_depth={best_max_depth}, "
      f"min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}")
print("验证集准确率：", best_fitness)

rf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth if best_max_depth > 0 else None,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    random_state=0
)
rf.fit(X_train, y_train)

# ---------------- 6. 评估与可视化 ----------------
y_pred = rf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("最终模型准确率:", acc)

# 混淆矩阵
cm = metrics.confusion_matrix(y_test, y_pred, labels=rf.classes_)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 特征重要性
feature_names = features
importances = rf.feature_importances_
plt.figure(figsize=(5,4))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.show()

# 示例树
plt.figure(figsize=(15, 10))
plot_tree(rf.estimators_[0], feature_names=feature_names,
          class_names=rf.classes_, filled=True, max_depth=4)
plt.title('Example Tree from Random Forest')
plt.show()

'''
    PCA（Principal Component Analysis） 是一种 降维方法，它通过线性组合把高维特征投影到较低维度，同时保留尽可能多的信息（方差）。
    换句话说，它做了两件事：
        去冗余 → 去掉相关性高的特征
        降低维度 → 用更少的特征表示数据

PCA 适用场景
场景 1：特征数量多（高维数据）
    特征列很多，例如上百或上千列
    可能存在 高度相关或冗余特征
    PCA 可以：
        压缩数据，减少计算量
        去掉冗余，提高模型训练速度
    例子：图像数据（每个像素是一个特征）、基因表达数据

场景 2：特征相关性强
    原始特征之间存在线性相关性
    PCA 可以：
        将相关特征 组合成独立的主成分，避免模型因为共线性（multicollinearity）不稳定
    例子：
        股票的 Open/High/Low/Close 数据
        某些经济指标可能高度相关

场景 3：可视化
    数据维度 > 2 或 3 时很难画图
    PCA 可以把数据降到 2 维或 3 维，用散点图查看数据分布或聚类趋势

场景 4：降噪
    PCA 保留高方差方向，舍弃低方差方向
    低方差方向可能是噪声
    降维可以提高模型鲁棒性
'''
'''
什么时候不一定需要 PCA?
    特征数量少（2~10 列） → 降维意义不大
    特征之间独立，冗余少 → PCA 效果有限
    模型对特征解释性要求高 → PCA 会把原始特征混合，不利于可解释性
'''
'''
怎么判断是否用 PCA?
    维度很高或有很多冗余特征 → 可以尝试 PCA
    特征间相关性高 → 可以检查相关系数矩阵
    高相关 → PCA 可以去冗余
    想做可视化 → 高维数据降到 2/3 维
    训练速度慢或模型过拟合 → 可以用 PCA 降维试试
'''
#%% 导入库
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%% 读取数据
file_name = "D:/MyProject/MatContest/dataset/raw/yf2014_2018.csv"
dataset = pd.read_csv(file_name)
dataset.head()

#%% 数据预处理
# 构建数值特征列
dataset['Open_N_num'] = np.where(dataset['Open'].shift(-1) > dataset['Open'], 1, 0)
dataset['Volume_N_num'] = np.where(dataset['Volume'].shift(-1) > dataset['Volume'], 1, 0)
dataset = dataset.dropna()

# 提取特征矩阵 X
# .values 将 DataFrame 转成 NumPy 数组，便于 PCA 运算
X = dataset[['Open', 'Open_N_num', 'Volume_N_num']].values

#%% 特征标准化
# 将每列标准化为 均值=0，标准差=1
# PCA 对特征尺度敏感，标准化保证每个特征在同一量级
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% PCA 降维
# 主成分数量 = 原始特征数量
# 这里只保留前两个主成分
# *** 这里不是说前两列特征 ***
# 而是PCA根据数据方差，找到前两个主成分方向（线性组合）
# 每个主成分 = 原始特征的加权组合
pca = PCA(n_components=2)# PCA 计算协方差矩阵 → 找到最大方差方向
# 计算协方差矩阵
# 求特征值、特征向量
# 把原始数据投影到前两个主成分方向
# 输出 X_pca → 每行数据在两个主成分上的坐标（降维后的新特征矩阵）
X_pca = pca.fit_transform(X_scaled)

# 假设找到了这两个主成分方向：[0.5, 0.7, 0.5] 和[-0.2, 0.4, -0.9]
# 由于 X_scaled 有三列：'Open', 'Open_N_num', 'Volume_N_num'
# 这里叫他们为x1,x2,x3
# 所以pca.fit_transform(X_scaled)就是在算
# 第一个主成分 PC1 = 0.5*x1 + 0.7*x2 + 0.5*x3
# 第一个主成分 PC1 = 0.5*x1 + 0.7*x2 + 0.5*x3


# 输出每个主成分的方差贡献率
# 用来判断 前两个主成分保留了多少原始信息
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("累计方差贡献率:", np.cumsum(pca.explained_variance_ratio_))

#%% 可视化前两个主成分
# 作用：可视化降维后的数据分布，查看是否存在聚类或趋势

plt.figure(figsize=(12,6))
# X_pca[:,0] → 第一个主成分坐标
# X_pca[:,0] → 第一个主成分坐标
# 蓝色，半透明
plt.scatter(X_pca[:,0], X_pca[:,1], c='blue', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - First 2 Principal Components')
plt.grid(True) # 添加网格
plt.show()

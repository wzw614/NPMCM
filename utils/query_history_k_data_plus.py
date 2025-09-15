'''
    获取日线数据
    用于跑回归模型
'''
import os
import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt

# 登陆系统
lg = bs.login()

# 获取数据
rs_result = bs.query_history_k_data_plus(
    "sh.600000",
    fields="date,open,high,low,close,preclose,volume,amount,adjustflag",
    start_date='2017-07-01',
    end_date='2017-12-31',
    frequency="d",
    adjustflag="3"
)
df_result = rs_result.get_data()

# 打印前5行看看数据结构
print(df_result.head())

# 确保路径存在
save_path = "../dataset/raw"
os.makedirs(save_path, exist_ok=True)

# 保存为 csv 文件
file_name = os.path.join(save_path, "sh600000_2017.csv")
df_result.to_csv(file_name, index=False, encoding="utf-8")
print(f"数据已保存到: {file_name}")

# 登出系统
bs.logout()

# ---------------- 可视化 ----------------
# 把日期列转为 datetime 类型
df_result["date"] = pd.to_datetime(df_result["date"])
df_result["close"] = df_result["close"].astype(float)

# 画收盘价曲线
plt.figure(figsize=(10, 5))
plt.plot(df_result["date"], df_result["close"], label="Close Price")
plt.title("SH600000 Close Price (2017)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


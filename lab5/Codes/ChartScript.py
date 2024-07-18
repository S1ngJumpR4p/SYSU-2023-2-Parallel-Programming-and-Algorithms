import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
filename = "2048.csv"
df = pd.read_csv(filename)

# 提取数据
threads = df.iloc[:, 0]
static_times = df.iloc[:, 1]
dynamic_times = df.iloc[:, 2]


# 绘制折线图
plt.plot(threads, static_times, marker='o', label='Static', color='red')
plt.plot(threads, dynamic_times, marker='o', label='Dynamic', color='blue')

# 添加标题和标签
plt.title(f"2048*2048")
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')

# 添加图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()

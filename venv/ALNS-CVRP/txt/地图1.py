import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件中的数据
with open('26.txt', 'r') as f:
    lines = f.readlines()

x = []
y = []

for line in lines:
    values = line.strip().split()
    x.append(float(values[0]))
    y.append(float(values[1]))

# 创建一个新的图形
plt.figure()

# 绘制点并连成线
for i in range(len(x)-1):
    plt.plot([x[i]-10, x[i+1]-10], [y[i], y[i+1]], linewidth=1)
    plt.plot([x[i]+10, x[i+1]+10], [y[i], y[i+1]], linewidth=1)

# # 使用 fill_between 函数填充线与坐标轴之间的区域
# for i in range(len(x)-1):
#     plt.fill_between([x[i]-10, x[i+1]-10], y[i], y[i+1], color='blue', alpha=0.2)
#     plt.fill_between([x[i]+10, x[i+1]+10], y[i], y[i+1], color='blue', alpha=0.2)

# 设置图形的标题和坐标轴标签
plt.title('坐标点连成线图（每条道路宽度为20）')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')

# 显示图形
plt.show()

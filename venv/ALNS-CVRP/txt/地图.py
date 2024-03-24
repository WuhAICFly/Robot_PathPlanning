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
print(x)
print(y)
# 创建一个新的图形
plt.figure()

# 绘制点
plt.plot(x, y)

# 设置图形的标题和坐标轴标签
#plt.title('坐标点图')
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 定义一个函数来生成8字形的坐标点
def eight_shape(t):
    x = np.sin(t)
    y = np.sin(t)*np.cos(t)
    return x, y  # 交换x和y的位置

# 生成坐标点
t = np.linspace(0, 2*np.pi, 100)
x, y = eight_shape(t)  # 交换x和y的位置

# 绘制8字形图形
plt.figure(1)  # 设置图形大小为100*100
plt.plot(50*(x+1), 50*(y+1))
x=50*(x+1)
y=50*(y+1)
x=np.round(x,2)
y=np.round(y,2)
print(x)
print(y)
# 添加标题和坐标轴标签
plt.title('竖向的8字形图形')
plt.xlabel('Y')  # x轴变为Y轴
plt.ylabel('X')  # y轴变为X轴

# 显示图形
plt.show()
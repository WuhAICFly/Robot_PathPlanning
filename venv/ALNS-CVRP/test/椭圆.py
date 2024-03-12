import matplotlib.pyplot as plt
import numpy as np

# 定义椭圆的参数
center = (50, 40)  # 椭圆中心点坐标
width = 40  # 椭圆宽度
height = 20  # 椭圆高度

# 计算椭圆上的点
t = np.linspace(0, 2*np.pi, 1000000)
x = center[0] + width/2 * np.cos(t)
y = center[1] + height/2 * np.sin(t)



# 绘制椭圆
plt.plot(x, y, label='Ellipse')



# 设置图像标题和坐标轴标签
plt.title('Ellipse')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图例
plt.legend()

# 显示图像
plt.show()
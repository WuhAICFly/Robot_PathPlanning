import matplotlib.pyplot as plt
import math

# 输入三个点的坐标
x1, y1 = input("请输入第一个点的坐标(x1,y1)：").split(',')
x2, y2 = input("请输入第二个点的坐标(x2,y2)：").split(',')
x3, y3 = input("请输入第三个点的坐标(x3,y3)：").split(',')

# 将输入的字符串转换为浮点数类型
x1, y1, x2, y2, x3, y3 = float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)

# 计算圆心坐标和半径
# 求出两条中垂线的交点(x0,y0)
k1 = (y2 - y1) / (x2 - x1)
k2 = (y3 - y2) / (x3 - x2)
x0 = (k1 * k2 * (y3 - y1) + k1 * (x2 + x3) - k2 * (x1 + x2)) / (2 * (k1 - k2))
y0 = -1 * (x0 - (x1 + x2) / 2) / k1 + (y1 + y2) / 2
r = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

# 输出圆心坐标和半径
print("圆心坐标为：({:.2f},{:.2f})".format(x0, y0))
print("半径为：{:.2f}".format(r))

# 画出圆
theta = [i * (2 * math.pi) / 1000 for i in range(1001)]
x = [r * math.cos(t) + x0 for t in theta]
y = [r * math.sin(t) + y0 for t in theta]

plt.plot(x, y)
plt.axis('equal')
plt.show()

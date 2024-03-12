import matplotlib.pyplot as plt
import numpy as np

# t = np.linspace(0, 2*np.pi, 50)
# x = 50*(np.sin(t)+1)
# y = 50*(np.sin(t)*np.cos(t)+1)
#
#
# x_int = (x).astype(int)
# y_int = (x_int-50)*np.sin(t+np.pi/2) +50
# print(x_int)
# print(y_int)
# for i in range(len(x_int)):
#     print("x: {}  y: {}".format(x_int[i], y_int[i]))

t = np.linspace(0, 2*np.pi, 50)
x = 50*(np.sin(t)+1)
y = 50*(np.sin(t)*np.cos(t)+1)

x_int = np.round(x).astype(int)
y_int = np.round(y, 3)
print(x_int)
print(y_int)
for i in range(len(x_int)):
    print("x =", x_int[i], "y =", y_int[i])

# 绘制8字形图形
plt.figure(1)  # 设置图形大小为100*100
plt.plot(x_int, y_int)
# 添加标题和坐标轴标签
plt.title('横向的8字形图形')
plt.xlabel('Y')  # x轴变为Y轴
plt.ylabel('X')  # y轴变为X轴

# 显示图形
plt.show()
# import numpy as np
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
#
# t = np.linspace(0, 2*np.pi, 100)
# x = 50*(np.sin(t)+1)
# y = 50*(np.sin(t)*np.cos(t)+1)
#
# y_int = np.round(y).astype(int)
# x_int = np.round(x).astype(int)
#
# x_new = np.linspace(0, 100, 101)
# y_new = interp1d(x_int, y_int, kind='linear')(x_new)
#
# plt.plot(x_new, y_new)
# plt.show()

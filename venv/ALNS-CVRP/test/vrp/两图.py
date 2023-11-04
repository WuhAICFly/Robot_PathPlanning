import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]

# 创建图表和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 在第一个子图中绘制第一个图形
ax1.plot(x, y1, label='y1')
ax1.set_xlabel('x轴')
ax1.set_ylabel('y轴')
ax1.set_title('第一个图形')
ax1.legend()

# 在第二个子图中绘制第二个图形
ax2.plot(x, y2, label='y2')
ax2.set_xlabel('x轴')
ax2.set_ylabel('y轴')
ax2.set_title('第二个图形')
ax2.legend()

# 显示图表
plt.show()
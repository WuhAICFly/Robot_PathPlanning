import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 确保已安装Times New Roman字体
font = FontProperties(fname='Times New Roman.ttf')

# 示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制图形
plt.plot(x, y)

# 设置图例和坐标轴标签
plt.legend(['Line 1'], prop=font)
plt.xlabel('X-axis', fontsize=16, fontproperties=font)
plt.ylabel('Y-axis', fontsize=16, fontproperties=font)

# 设置坐标轴标签字体
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)

# 显示图形
plt.show()
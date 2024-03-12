import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 24})

# 生成数据
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# 绘制图形
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Function')
plt.legend()
plt.show()

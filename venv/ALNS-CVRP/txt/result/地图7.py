import matplotlib.pyplot as plt
import matplotlib.image as imread
from matplotlib.font_manager import FontProperties
import numpy as np
from typing import List, Tuple
import math
import time



# # 指定中文字体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 指定英文字体
plt.rc('font',family='Times New Roman')


road=[(238, 106),(126,178),(184, 291),(104, 329),(184, 291),(126,178),(238, 106)]
def readtxt(filename):
  with open(filename, "r") as file:
    lines = file.readlines()
    x1 = []
    y1 = []
    for line in lines:
        index, number = line.strip().split()
        x1.append(index)
        y1.append(number)
  x1=np.array(x1,dtype=np.float64)
  y1=np.array(y1,dtype=np.float64)
  return x1, y1

x1,y1=readtxt('S_QP.txt')

print(x1,y1)
# 读取图片
image_path = 'image1.png'
img = imread.imread(image_path)

# 将图片水平翻转
img = np.fliplr(img)
# 将图片旋转180度
img = np.rot90(img, 2)
# 显示图片
plt.imshow(img, cmap='gray')

start = road[0]
end = road[-1]
# 设置全局字体大小
plt.rcParams['font.size'] = 12
# 将点依次连接并画出
ix=[]
iy=[]
for i in range(len(road)-1):
    ix.append(road[i][0])
    iy.append(road[i][1])
plt.plot(ix,iy,label='Initial path',color='b',linewidth=1)

plt.plot(x1, y1,label='Tracking path', linestyle='--', color='r')
# 设置标题
# plt.title('Image')

plt.tick_params(axis='both', labelsize=14)
# 显示坐标轴
plt.axis('on')
plt.xlabel("x/m",fontdict={'size': 20})
plt.ylabel("y/m",fontdict={'size': 20})
# 绘制两点之间的线
#plt.plot([63, 221], [281, 171], color='red', linewidth=2)

# 反转y轴，使0刻度位于左下方
plt.gca().invert_yaxis()

# 设置y轴范围，使0刻度位于左下方
#plt.xlim([387000, 388000])
#plt.ylim([3110460, 3111520])

# 显示图像
plt.legend(loc='upper right',fontsize=12)
# 显示图片
plt.show()

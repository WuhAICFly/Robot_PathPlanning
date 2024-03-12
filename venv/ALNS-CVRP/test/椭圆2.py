import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from typing import List, Tuple
import math
import time
plt.rc('font',family='Times New Roman')


# 定义函数
def calculate_y(x):
    return np.sqrt(400*(1 - (x-20)**2/400))+30
def calculate_Ny(x):
    return -np.sqrt(400*(1 - (x-20)**2/400))+30
fig = plt.figure(1)
xx=[]
yy=[]
xy=[]
# 生成 x 的取值范围
x = np.arange(0, 41, 0.01)
# 计算对应的 y 值
y = calculate_y(x)
Ny=calculate_Ny(x)
xy=zip(x,y)
xNy=zip(x,Ny)

with open("output.txt", "a") as file:
    for index, number in enumerate(Ny, start=120):
        file.write(f"{index} {number}\n")

with open("S_QP.txt", "r") as file:
    lines = file.readlines()
    x1 = []
    y1 = []
    for line in lines:
        index, number = line.strip().split()
        x1.append(index)
        y1.append(number)
x1=np.array(x1,dtype=np.float64)
y1=np.array(y1,dtype=np.float64)
for i in range(801,1601):
    x1[i]=160-x1[i]
print("x:", len(x1))
print("y:", len(y1))

# 绘制图形
#fig, ax = plt.subplots()
plt.rcParams['font.size'] = 20
plt.plot(x, y,'b',label='Initial path')
plt.plot(x, Ny,'b')
plt.plot(x+40, y,'b')
plt.plot(x+40, Ny,'b')
plt.plot(x1, y1,'r--',label='Tracking path')
#plt.title('Plot of y = sqrt(1 - x**2/400)')
plt.xlabel("x/m",fontdict={'size': 20})
plt.ylabel("y/m",fontdict={'size': 20})
# 设置全局字体大小
#plt.rcParams.update({'font.size': 12})

# 显示图像
plt.legend(loc='upper right',fontsize=12)  # 调整标签位置为右上角


#plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from typing import List, Tuple
import math
import time

#import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimSun' ,'Times New Roman']  # 指定字体，支持中文和英文
#plt.rc('font',family='Times New Roman')

# # 定义函数
# def calculate_y(x):
#     return np.sqrt(400*(1 - (x-20)**2/400))+30
# def calculate_Ny(x):
#     return -np.sqrt(400*(1 - (x-20)**2/400))+30
# fig = plt.figure(1)
# xx=[]
# yy=[]
# xy=[]
# # 生成 x 的取值范围
# x = np.arange(0, 41, 0.01)
# # 计算对应的 y 值
# y = calculate_y(x)
# Ny=calculate_Ny(x)
# xy=zip(x,y)
# xNy=zip(x,Ny)
#
# with open("output.txt", "a") as file:
#     for index, number in enumerate(Ny, start=120):
#         file.write(f"{index} {number}\n")

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
# for i in range(801,1601):
#     x1[i]=160-x1[i]
  return x1,y1

#x1,y1=readtxt('S_QP.txt')
x2,y2=readtxt('E_QP.txt')
x3,y3=readtxt('v_QP.txt')
x4,y4=readtxt('a_QP.txt')
x5,y5=readtxt('k_QP.txt')

# 绘制图形
#fig, ax = plt.subplots()

# plt.plot(x, y,'b',label='Initial path')
# plt.plot(x, Ny,'b')
# plt.plot(x+40, y,'b')
# plt.plot(x+40, Ny,'b')
# 设置全局字体大小
plt.rcParams['font.size'] = 20
#plt.plot(x1, y1,label='S_QP')
plt.plot(x2, y2,label='E_QP',linewidth=2)
plt.plot(x3, y3,label='v_QP',linewidth=2)
plt.plot(x4, y4,label='a_QP',linewidth=2)
plt.plot(x5, y5,label='k_QP',linewidth=2)
# 中文
plt.xlabel("时间/秒",fontdict={'size': 20})
plt.ylabel("跟踪误差(m) 速度(m/s) 加速度(m/s²) 曲率",fontdict={'size': 20})
#英文
# plt.xlabel("Time/s",fontdict={'size': 20})
# plt.ylabel("Tracking error(m) Velocity(m/s) Acceleration(m/s²) Curvature",fontdict={'size': 12})
# 显示图像
plt.legend(loc='upper right',fontsize=12)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from typing import List, Tuple
import math
import time
import  os
from matplotlib.font_manager import FontProperties
# 设置y轴标签的字体大小
font_properties = FontProperties(fname='timessimsun.ttf')  # 指定中文字体




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
#画线
def ddraw(val,lag):
    if val == 0:
     plt.plot(x2, y2,linewidth=2)
     if lag==0:
         plt.xlabel("时间/秒",fontproperties=font_properties,size=16)
         plt.ylabel("跟踪误差(m)",fontproperties=font_properties,size=16)
     else:
         plt.xlabel("Time/s",fontproperties=font_properties,size=16)
         plt.ylabel("Tracking error(m)",fontproperties=font_properties,size=16)
    if val == 1:
     plt.plot(x3, y3,linewidth=2)
     if lag==0:
         plt.xlabel("时间/秒", fontproperties=font_properties,size=16)
         plt.ylabel("速度(m/s)", fontproperties=font_properties,size=16)
     else:
         plt.xlabel("Time/s",fontproperties=font_properties,size=16)
         plt.ylabel("Velocity(m/s)",fontproperties=font_properties,size=16)
    if val == 2:
     plt.plot(x4, y4,linewidth=2)
     if lag==0:
         plt.xlabel("时间/秒",fontproperties=font_properties,size=16)
         plt.ylabel("加速度(m/s²)", fontproperties=font_properties,size=16)
     else:
         plt.xlabel("Time/s", fontproperties=font_properties,size=16)
         plt.ylabel("Acceleration(m/s²)",fontproperties=font_properties,size=16)
    if val == 3:
     plt.plot(x5, y5,linewidth=2)
     if lag==0:
         plt.xlabel("时间/秒",fontproperties=font_properties,size=16)
         plt.ylabel("曲率",fontproperties=font_properties,size=16)
     else:
         plt.xlabel("Time/s",fontproperties=font_properties,size=16)
         plt.ylabel("Curvature",fontproperties=font_properties,size=16)

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
plt.rcParams['font.size'] = 15
#plt.plot(x1, y1,label='S_QP')
i=0
j=0
# 指定保存图片的位置

for j in range(2):
 # # 创建文件夹
 # save_path = f'C:/Users/wuhon/Desktop/'  # 请将此路径更改为您想要保存图片的实际路径
 # os.makedirs(save_path+f'{j}', exist_ok=True)
 # save_path = f'C:/Users/wuhon/Desktop/{j}'  # 请将此路径更改为您想要保存图片的实际路径
 # # 改变当前的工作目录到指定的文件夹
 # os.chdir(save_path)
 for i in range(4):
    # 创建一个图形
  plt.figure()
  ddraw(i,j)
  #plt.show()
# 指定图片的尺寸（分辨率）
  fig = plt.gcf()
  fig.set_size_inches(10, 6)  # 设置图像的宽度和高度（单位为英寸）
  # 保存图片
  plt.savefig(f'{j}{i}.png',dpi=100)
  # 关闭图形
  plt.close()
  i+=1;
#language(0,'速度')
# 显示图像
#plt.legend(loc='upper right',fontsize=12)

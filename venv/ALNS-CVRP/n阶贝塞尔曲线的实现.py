# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 21:33:32 2023

@author: 86136
"""

## 递归的方式实现贝塞尔曲线
import numpy as np
import matplotlib.pyplot as plt
## 递归的方式实现贝塞尔曲线
def bezier(Ps,n,t):
    """递归的方式求解贝塞尔点

    Args:
        Ps (_type_): 控制点，格式为numpy数组：array([[x1,y1],[x2,y2],...,[xn,yn]])
        n (_type_): n个控制点，即Ps的第一维度
        t (_type_): 时刻t

    Returns:
        _type_: 当前t时刻的贝塞尔点
    """
    if n==1:
        return Ps[0]
    return (1-t)*bezier(Ps[0:n-1],n-1,t)+t*bezier(Ps[1:n],n-1,t)

# 画图验证
Ps = np.array([[0,0],[1,1],[2,1],[3,0],[3,1],[4,2],[5,2],[6,8]])
print(Ps)

x_=[]
y_=[]
#for t in np.arange(0,1,0.01):
plt.cla()
pos = bezier(Ps,len(Ps),0.1)
#print(pos[0])
x_.append(pos[0])
y_.append(pos[1])
plt.plot(Ps[:,0],Ps[:,1])
plt.scatter(x_,y_,c='r',s=1)
# print(pos)
# plt.plot(pos[0],pos[1])
plt.pause(0.001)
plt.show()



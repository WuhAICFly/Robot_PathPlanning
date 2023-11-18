# -*- coding: utf-8 -*-
"""
贪心法
name:JCH
date:6.8
"""
import pandas as pd
import numpy as np
import math
import time
import csv


def readtxt(i):
    with open('../tsp1.txt', 'r') as f:
        lines = f.readlines()
        points = lines[i]
        ppoints = eval(points)
        points = [p[0] for p in ppoints]
        del points[0]
        # draw(points)
        # 将元组转换为列表
        lst = []
        for p in points:
            p = [i for i in p]
            lst.append(p)
        #print(lst)
        return lst , ppoints
def write_to_file(lst):
    with open('output.tsp', 'w') as tspfile:
        for i, row in enumerate(lst):
            tspfile.write(f"{i} {row[0]} {row[1]}\n")

lst,ppoints=readtxt(9)
print(lst)
write_to_file(lst)
dataframe = pd.read_csv("output.tsp",sep=" ",header=None)
v = dataframe.iloc[:,1:3]
print("v:",v)
train_v= np.array(v)
train_d=train_v
dist = np.zeros((train_v.shape[0],train_d.shape[0]))

#计算距离矩阵
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
"""
s:已经遍历过的城市
dist：城市间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
flag：访问标记
"""
i=1
n=train_v.shape[0]
j=0
sumpath=0
s=[]
s.append(0)
start = time.clock()
while True:
    k=1
    Detemp=10000000
    while True:
        l=0
        flag=0
        if k in s:
            flag = 1
        if (flag==0) and (dist[k][s[i-1]] < Detemp):
            j = k;
            Detemp=dist[k][s[i - 1]];
        k+=1
        if k>=n:
            break;
    s.append(j)
    i+=1;
    sumpath+=Detemp
    if i>=n:
        break;
sumpath+=dist[0][j]
end = time.clock()
print("结果：")
print(ppoints)
print(sumpath)
with open("data.txt", "w") as f:
    f.write(str(ppoints) + "\n")
    f.write(str(sumpath) + "\n")
for m in range(n):
    print("%s "%(s[m]))
print("程序的运行时间是：%s"%(end-start))
"""
结果：
10464.1834865
0 
9 
8 
5 
3 
2 
1 
7 
4 
6 
程序的运行时间是：7.724798388153431e-05
"""
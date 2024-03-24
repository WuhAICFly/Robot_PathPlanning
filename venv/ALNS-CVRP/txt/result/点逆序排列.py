import numpy as np
import os
# 指定文件路径
num=4
filename='74'
file='v_QP'
myfilename='05'

flag=0   #0 偏移 1 y逆序  3 y逆序+偏移    2 点逆序
value=324
#file_path = 'J:\\PyCharm Community Edition with Anaconda plugin 2020.2.5\\pythonProject2\\venv\\ALNS-CVRP\\txt\\result\\00\\a_QP.txt'
file_path = f"J:\\PyCharm Community Edition with Anaconda plugin 2020.2.5\\pythonProject2\\venv\\ALNS-CVRP\\txt\\result\\{filename}/{file}.txt"



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

def Yreverseodder(x,y):

    # # 对点进行逆序排列
    # points.reverse()
    # 反转y的顺序
    y = y[::-1]
    points = list(zip(x, y))
    # 打印逆序排列后的点
    print(points)
    # for i in range(len(points) - 1):
    #     print(f"{points[i][0]}  {points[i][1]}")
    # 指定目录和文件名
    directory = f"J:\\PyCharm Community Edition with Anaconda plugin 2020.2.5\\pythonProject2\\venv\\ALNS-CVRP\\txt\\result\\{myfilename}\\"
    myfile = f"{file}{num}.txt"
    file_path = os.path.join(directory, myfile)

    # 写入文件
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f'{point[0]} {point[1]}\n')
def Preverseodder(pos):

    # # 对点进行逆序排列
    pos.reverse()
    # 打印逆序排列后的点
    print(pos)
    # for i in range(len(points) - 1):
    #     print(f"{points[i][0]}  {points[i][1]}")
    # 指定目录和文件名
    directory = f"J:\\PyCharm Community Edition with Anaconda plugin 2020.2.5\\pythonProject2\\venv\\ALNS-CVRP\\txt\\result\\{myfilename}\\"
    myfile = f"{file}{num}.txt"
    # 创建文件夹
    if not os.path.exists(directory):
        os.makedirs(directory)
    # fold_path = os.path.join(directory, myfilename)
    # if not os.path.exists(fold_path):
    #     os.makedirs(fold_path)
    file_path = os.path.join(directory, myfile)
    # 写入文件
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f'{point[0]} {point[1]}\n')
def  offs(data,value):
    shifted_data = [(x + value, y) for x, y in data]
    print(shifted_data)

    # 指定目录和文件名
    directory = f"J:\\PyCharm Community Edition with Anaconda plugin 2020.2.5\\pythonProject2\\venv\\ALNS-CVRP\\txt\\result\\{myfilename}\\"
    myfile = f"{file}{num}.txt"

    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, myfile)
    # 写入文件
    with open(file_path, 'w') as f:
        for point in shifted_data:
            f.write(f'{point[0]} {point[1]}\n')
x,y=readtxt(file_path)

points = list(zip(x, y))
print(points)

if flag==1:
  Yreverseodder(x,y)
elif flag==0:
   offs(points,value)
elif flag==2:
   Preverseodder(points)
else:
    Yreverseodder(x,y)
    offs(points,value)

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera  # 保存动图时用，pip install celluloid
from queue import PriorityQueue
from math import sqrt
show_animation = True
def ReadObstacle(i, list2, list3, times, dtax, dtay):
    filename = str(i) + "." + "txt"
    # 打开文件
    with open(filename, 'r') as f:
        # 定义两个空列表存储提取的数据
        # 循环遍历每一行数据
        for line in f.readlines():
            # 去除每行末尾的空格和换行符
            line = line.strip()
            # 使用空格分隔每行数据
            data = line.split(' ')
            print(data)
            # 将第2个和第3个数据分别存储到对应的列表中
            list2.append(float(data[0]) * times - dtax)
            list3.append(float(data[1]) * times + dtay)

def main(rng=None):
    print(" start!!")
    fig = plt.figure(1)

    # camara = Camera(fig)  # 保存动图时使用
    camara = None
    ox = []
    oy = []
    j=26
    for i in range(1):
        ReadObstacle(j, ox, oy, 1, 0, 0)
        xy=list(zip(ox,oy))
        print(xy)
        if show_animation:
            #plt.plot(ox, oy, ".k")
            plt.plot(ox, oy,'gray')
            plt.axis("equal")
            if camara != None:
                camara.snap()
        j=j+1;

    plt.show()

if __name__ == '__main__':
    main()
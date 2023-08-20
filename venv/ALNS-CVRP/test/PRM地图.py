import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera  # 保存动图时用，pip install celluloid
show_animation = True

def opentxt():
 n=0
 sum=0
 allsum=0
 for n in range(22):
    #filename = '../txt/y_values{}'.format(n)  # py生成文件名
    filename = '../Otxt/pathvalues{}.txt'.format(n)#matlab

    # 读取txt中的数据
    #data = np.loadtxt(filename, delimiter=",")#py
    data = np.loadtxt(filename)#matlab
    # 将数据分离为x和y坐标
    x_list = data[:, 0]
    y_list = data[:, 1]
    #point_list = list(zip(x, y))
    # 计算长度
    length = 0
    for i in range(len(x_list) - 1):
        x1, y1 = x_list[i], y_list[i]
        x2, y2 = x_list[i + 1], y_list[i + 1]
        length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    print(length)
    #绘制坐标图

    if n<4:
      plt.plot(x_list, y_list, 'r')
      sum+=length
      if n==3:
        print("老线路0路段长度：",sum)
        allsum += sum
        sum=0
    if 4<=n<6:
      plt.plot(x_list, y_list, 'g')
      sum += length
      if n == 5:
          print("老线路1路段长度：", sum)
          allsum += sum
          sum = 0
    if 6<=n < 8:
      plt.plot(x_list, y_list, 'y')
      sum += length
      if n == 7:
          print("老线路2路段长度：", sum)
          allsum += sum
          sum = 0
    if 8<=n < 11:
      plt.plot(x_list, y_list, 'b')
      sum += length
      if n == 10:
          allsum += sum
          print("老线路3路段长度：", sum)
          print("老线路长度：", allsum)
          sum = 0
          allsum=0

    if 11<=n < 15:
     plt.plot(x_list, y_list, 'r')
     sum += length
     if n == 14:
         allsum += sum
         print("新线路0路段长度：", sum)
         sum = 0
    if 15<=n < 20:
     plt.plot(x_list, y_list, 'g')
     sum += length
     if n == 19:
         allsum += sum
         print("新线路1路段长度：", sum)
         sum = 0
    if 20<=n < 22:
     plt.plot(x_list, y_list, 'y')
     sum += length
     if n == 21:
         allsum += sum
         print("新线路2路段长度：", sum)
         print("新线路长度：", allsum)
         sum = 0
         allsum=0


 # plt.show()


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
            # 将第2个和第3个数据分别存储到对应的列表中
            list2.append(float(data[1]) * times - dtax)
            list3.append(float(data[2]) * times + dtay)

def main(rng=None):
    print(" start!!")
    fig = plt.figure(1)

    # camara = Camera(fig)  # 保存动图时使用
    camara = None
    # start and goal position
    sx = -5.0  # [m]
    sy = 10.0  # [m]
    gx = 25.0  # [m]
    gy = 5.0  # [m]
    robot_size = 1.3  # [m]

    ox = []
    oy = []
    x = []
    y = []
    points = [(-5, 10), (33.2, 9.9), (35.2, 20.5), (39.9, 34.0), (51.4, -1.1), (-9.3, 8.9), (30, 10), (33, 13)]
    #fig, ax = plt.subplots()
    plt.scatter([p[0] for p in points], [p[1] for p in points], marker='^', s=60)

    for i, p in enumerate(points):
        plt.annotate(str(i), (p[0] + 0.7, p[1] - 0.5))
    #plt.show()
    #6 外圈障碍物
    ReadObstacle(6, ox, oy, 1.7, 8, 5)
    p1 = np.array([7.59297399782e-05, 5.93557160755e-07])
    p2 = np.array([-2.49960255623, -1.03229379654])
    points = np.linspace(p1, p2, num=20, endpoint=True)
    # 遍历每个点并将其x、y坐标加入列表中
    for point in points:
        ox.append(point[0] * 1.7 - 8)
        oy.append(point[1] * 1.7 + 5)
    # 1-4障碍物
    for i in range(6):
        ReadObstacle(i, ox, oy, 1, 0, 0)
    # 7障碍物
    ReadObstacle(7, ox, oy, 1.2, -20, 0)
    opentxt()
    if show_animation:
        plt.plot(ox, oy, ".k")


        plt.axis("equal")
        if camara != None:
            camara.snap()
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    main()
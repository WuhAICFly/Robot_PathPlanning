import numpy as np
from vrp import *
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def write_to_file(file_name, data):
 with open(file_name, 'a') as f:
  data = str(data)
  f.write(data)
  f.write(' ')
  # f.write('\n')
def draw(path1,locations):
    #路线图绘制
    fig=plt.figure(1)
    for path in path1:
        plt.plot(locations[path][:,0],locations[path][:,1], marker='o')

    plt.show()

with open('../path', 'r') as f:
   path = f.read()
path = eval(path)  # 将字符串转换为列表格式
#print(path)

# path = [[0, 26, 54, 55, 25, 24, 29, 68, 80, 0], [0, 50, 81, 79, 3, 77, 12, 0], [0, 1, 30, 20, 66, 71, 9, 33, 0],
#  [0, 88, 62, 10, 63, 90, 32, 70, 69, 27, 0], [0, 28, 76, 78, 34, 35, 65, 51, 31, 0], [0, 52, 82, 48, 8, 45, 83, 18, 0],
#  [0, 6, 61, 16, 86, 17, 84, 60, 89, 0], [0, 94, 98, 91, 44, 100, 37, 92, 95, 0], [0, 59, 97, 87, 13, 0],
#  [0, 75, 22, 41, 15, 43, 38, 14, 42, 57, 2, 0], [0, 58, 21, 72, 74, 73, 40, 53, 0], [0, 4, 39, 67, 23, 56, 0],
#  [0, 7, 19, 11, 64, 49, 36, 47, 46, 0], [0, 5, 85, 93, 99, 96, 0]]

#路段数据整合
lst1, lst2=vrp()
locations = np.array(lst1)
lst3 = list(zip(lst1, lst2))
print(lst3)
pos = []
for sublist in path:
    temp = []
    for index in sublist:
        temp.append(lst3[index])
    pos.append(temp)
print(pos)
draw(path,locations)
kdtree = KDTree(np.vstack((lst1[1:][0], lst1[1:][1])).T)
dist, index = kdtree.query([lst1[0][0], lst1[0][1]],k=1)




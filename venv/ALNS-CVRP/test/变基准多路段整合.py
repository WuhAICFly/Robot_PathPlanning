import numpy as np
from vrp import *
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from typing import List, Tuple
import math
def write_to_file(file_name, data):
 with open(file_name, 'a') as f:
  data = str(data)
  f.write(data)
  f.write(' ')
  # f.write('\n')
def draw(path1,locations,points,minpath):
    #路线图绘制
    fig=plt.figure(1)
    for path in path1:
        plt.plot(locations[path][:,0],locations[path][:,1], marker='o')
    plt.scatter([p[0] for p in points], [p[1] for p in points], marker='^', s=60)
    for i, p in enumerate(points):
        plt.annotate(str(i+1), (p[0] + 0.7, p[1] - 0.5))
    p = [p[0] for p in minpath]
    for pos in p:
        plt.plot(pos[:][0], pos[:][1],marker='*')
    # 设置轴范围
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()
def pathdistance(lst,pos,k):
    # 初始化距离之和为0
    total_distance = 0
    points=[]
    # 遍历列表中的每个点
    n=0
    for point, _ in lst:
        # 计算点到原点的距离
        points.append(point)
        distance = math.sqrt((point[0] - pos[0]) ** 2 + (point[1] - pos[1]) ** 2)
        # 将距离加入总距离
        n+=1
        total_distance += distance
    average=total_distance/n
    # plt.plot(points[:][0],points[:][1],color='black')
    # 打印总距离
    print("距离:",n,average)
    return n,average,k
def nearest_k_points(x_values, y_values):
    kdtree = KDTree(np.vstack((x_values, y_values)).T)
    dist, index = kdtree.query([lst1[0][0],lst1[0][1] ],k=10)
    for i in range(len(index)):
        indx = index[i]
        d.append(dist[i])
        nearest_k_points.append(kdtree.data[indx])
    nearest_k_points = [(d[0], d[1]) for d in nearest_k_points]
    # print(index)
    print("nearest_k_points:",nearest_k_points)
    print("current point:",lst1[0])
    return nearest_k_points
def find_k_nearest_neighbors(points: List[Tuple[float, float]], target_point: Tuple[float, float], k: int) -> List[
    Tuple[Tuple[float, float], float]]:
    # 构建kdtree
    tree = KDTree(points)

    # 找k个最近邻
    dist, ind = tree.query([target_point], k=k)
    #print(ind)
    # print("nearest_k_points:", nearest_k_points)
    # 返回k个最近邻的坐标和距离
    return [(tuple(tree.data[i]), d) for i, d in zip(ind[0], dist[0])],ind
#找距离平均最小路段
def foundPath(pos,piont):
    k = -1
    averageNum = []
    kNum = []
    for p in pos:
        k += 1
        n, average, k = pathdistance(p, piont, k)
        averageNum.append(average)
        kNum.append(k)
    min_val = min(averageNum)
    min_idx = averageNum.index(min_val)
    print(min_idx)
    return min_idx

def findindex(target,pos):
    #target = (2.0, 60.0)
    for i, item in enumerate(pos):
        if target in [x[0] for x in item]:
            return i
    else:
        return -1
capacity=112

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
print("带需求路径点lst3",lst3)
pos = []
for sublist in path:
    temp = []
    for index in sublist:
        temp.append(lst3[index])
    pos.append(temp)
print("路段pos",pos)
print("pos段数",len(pos))
##求最近K近邻
# d=[]
# nearest_k_points=[]
# print("除出发点系列lst1[1:]",lst1[1:])
# x_values = [item[0] for item in lst1[1:]]
# y_values = [item[1] for item in lst1[1:]]
# kdtree = KDTree(np.vstack((x_values, y_values)).T)
# dist, index = kdtree.query([lst1[0][0],lst1[0][1] ],k=10)
# for i in range(len(index)):
#     indx = index[i]
#     d.append(dist[i])
#     nearest_k_points.append(kdtree.data[indx])
# nearest_k_points = [(d[0], d[1]) for d in nearest_k_points]
# # print(index)
# print("nearest_k_points:",nearest_k_points)
# print("current point:",lst1[0])
#points = [(41, 49), (35, 17), (55, 45), (55, 20), (15, 30), (25, 30), (20, 50), (10, 43), (55, 60), (30, 60), (20, 65), (50, 35), (30, 25), (15, 10), (30, 5), (10, 20), (5, 30), (20, 40), (15, 60), (45, 65), (45, 20), (45, 10), (55, 5), (65, 35), (65, 20), (45, 30), (35, 40), (41, 37), (64, 42), (40, 60), (31, 52), (35, 69), (53, 52), (65, 55), (63, 65), (2, 60), (20, 20), (5, 5), (60, 12), (40, 25), (42, 7), (24, 12), (23, 3), (11, 14), (6, 38), (2, 48), (8, 56), (13, 52), (6, 68), (47, 47), (49, 58), (27, 43), (37, 31), (57, 29), (63, 23), (53, 12), (32, 12), (36, 26), (21, 24), (17, 34), (12, 24), (24, 58), (27, 69), (15, 77), (62, 77), (49, 73), (67, 5), (56, 39), (37, 47), (37, 56), (57, 68), (47, 16), (44, 17), (46, 13), (49, 11), (49, 42), (53, 43), (61, 52), (57, 48), (56, 37), (55, 54), (15, 47), (14, 37), (11, 31), (16, 22), (4, 18), (28, 18), (26, 52), (26, 35), (31, 67), (15, 19), (22, 22), (18, 24), (26, 27), (25, 24), (22, 27), (25, 21), (19, 21), (20, 26), (18, 18)]
#求K近邻
#nearest_neighbors,ind1 = find_k_nearest_neighbors(lst1[1:], lst1[0],2)
for pathes in pos:
    x=[x[0] for x in pathes]
    print(x)
    nearest_neighbors, ind1 = find_k_nearest_neighbors(x, lst1[0], 3)

    print("离出发点最近nearest_neighbors:", nearest_neighbors)
k_neighbors = [x[0] for x in nearest_neighbors]
print(k_neighbors)
#求最近邻所在路段索引
id=findindex(k_neighbors[2],pos)
print("id:",id)
print("路点在路段pos[id]",pos[id])
my_list=pos[id]
#删除第id项，即第id项路段
del pos[id]
##求距离平均最小路段 返回最小路段
minpathindex=foundPath(pos,k_neighbors[0])
p=[p[0] for p in pos[minpathindex]]
print("最小距离路段：",p)
print("带需求最小距离路段：",pos[minpathindex])
neighbors,ind2 = find_k_nearest_neighbors(p, k_neighbors[0],len(p))
n_neighbors = [x[0] for x in neighbors]
print("最小距离路段中离出发点最近邻：",n_neighbors)
print("最小距离路段中离出发点带距离最近邻：",neighbors)
reorder_p = sorted(pos[minpathindex], key=lambda x: next((i[1] for i in neighbors if i[0] == x[0]), None))
print("带需求最小距离路段重排reorder_p：",reorder_p)

q=[q[1] for q in my_list]
total = sum(q)
num=(ind1[0][0]+1)
print(num)
Free_capacity=capacity-total+ (lst3[num][1]/2)
print("容量和：",Free_capacity)
goal=[]
for i in range(2, len(reorder_p)):
     if reorder_p[i][1]<Free_capacity:
         goal.append(reorder_p[i][1]/neighbors[i][1])#容量尽量大，距离尽量小
     else: break
print(goal)
if goal!=[]:
   maxgoal=max(goal)
   index = goal.index(maxgoal)+2
   print(maxgoal,index)

#把最近邻加入路段

for neighborindex, item in enumerate(my_list):
    if item[0] == (37.0, 31.0):
        print("第一次出现相同项的索引是：", neighborindex)
        break
print("索引",neighborindex)
new_lst = my_list[:neighborindex] + [(my_list[neighborindex][0], my_list[neighborindex][1] / 2.0)] + my_list[neighborindex+1:]
new_lst.append(reorder_p[index])
print("新的TSP路段:",new_lst)
write_to_file('tsp1.txt',new_lst)











#画图
draw(path,locations,k_neighbors,pos[minpathindex])


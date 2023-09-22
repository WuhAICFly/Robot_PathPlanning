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
  f.write('\n')
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
    #tree = KDTree(points)
    tree = KDTree(np.asarray(points, dtype=object))
    # 找k个最近邻
    dist, ind = tree.query([target_point], k=k)
    print(ind)
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

def tspPos(ind,flag,currentPath,ppos,pos):
    print("当前路段currentPath:", currentPath)
    # flag 相邻段能加入当前段flag=1，否则为0
    if flag == 1:  # 如果上一次发生拆分
       del currentPath[ind]  # 删除合并点
       ppos = [i for x in ppos for i in x]
       currentPath.append(tuple(ppos))  # 加入拆分点
    curPath = [p[0] for p in currentPath]
    neighbors, ind2 = find_k_nearest_neighbors(curPath, lst1[0], len(curPath))  # 从当前路段找到离原点最近点作为拆分点
    print("离出发点最近neighbors:", neighbors[2][0])
    splittpoint = neighbors[2][0]  # 拆分点
    currentPath = currentPath
    # 求最近邻所在路段索引
    id = findindex(neighbors[2][0], pos)
    print(id)
    # 删除第id项，即第id项路段
    del pos[id]
    curroad = pos
    ##求距离平均最小路段 返回最小路段
    print("curroad:", curroad)
    minpathindex = foundPath(curroad, neighbors[2][0])
    p = [p[0] for p in curroad[minpathindex]]
    print("最小距离路段：", p)

    print("带需求最小距离路段：", curroad[minpathindex])
    neighbors, ind3 = find_k_nearest_neighbors(p, neighbors[2][0], len(p))
    n_neighbors = [x[0] for x in neighbors]
    print("最小距离路段中离出发点最近邻：", n_neighbors)
    print("最小距离路段中离出发点带距离最近邻：", neighbors)
    reorder_p = sorted(curroad[minpathindex], key=lambda x: next((i[1] for i in neighbors if i[0] == x[0]), None))
    print("带需求最小距离路段重排reorder_p：", reorder_p)
    # 求满足容量限制的最优合并点
    print(currentPath)
    q = [q[1] for q in currentPath]
    total = sum(q)
    print(total)
    num = lst1.index(splittpoint)
    print(num)  # 拆分点索引号
    Free_capacity = capacity - total + (lst3[num][1] / 2)
    print("容量和：", Free_capacity)
    goal = []
    # 求出满足容量限制最优合并点
    for i in range(2, len(reorder_p)):
        if reorder_p[i][1] < Free_capacity:
            goal.append(reorder_p[i][1] / neighbors[i][1])  # 容量尽量大，距离尽量小
        else:
            break
    print("goal:", goal)
    if goal != []:
        flag = 1
        maxgoal = max(goal)
        index = goal.index(maxgoal) + 2
        print(maxgoal, index)
        # 把最近邻加入路段
        for neighborindex, item in enumerate(currentPath):
            if item[0] == splittpoint:
                print("第一次出现相同项的索引是：", neighborindex)
                break
        print("索引", neighborindex)
        new_lst = currentPath[:neighborindex] + [
            (currentPath[neighborindex][0], currentPath[neighborindex][1] / 2.0)] + currentPath[neighborindex + 1:]
        ppos = [(currentPath[neighborindex][0], currentPath[neighborindex][1] / 2.0)]  # 拆分需求的拆分点

        new_lst.append(reorder_p[index])
        print("新的TSP路段:", new_lst)
        write_to_file('tsp1.txt', new_lst)


    else:
        flag = 0
        index=0
        ppos=lst3[0]
    currentPath = curroad[minpathindex]#下一个路段
    ind = currentPath.index(reorder_p[index])#上一次合并点索引号

    return ind,flag,currentPath,ppos,curroad,splittpoint

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
print("带需求路径点lst1",lst1)
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
nearest_neighbors,ind1 = find_k_nearest_neighbors(lst1[1:], lst1[0],2)
print("离出发点最近nearest_neighbors:", nearest_neighbors)
k_neighbors = [x[0] for x in nearest_neighbors]
print(k_neighbors)
#求最近邻所在路段索引
id=findindex(k_neighbors[0],pos)
print(id)
print("路点在路段pos[id]",pos[id])#确定拆分起始路段
flag=0
minpathindex=0
currentPath=pos[id]
#currentPath=[((35.0, 35.0), 0.0), ((21.0, 24.0), 28.0), ((25.0, 21.0), 12.0), ((28.0, 18.0), 26.0), ((30.0, 25.0), 23.0), ((35.0, 35.0), 0.0)]
ind=-1
ppos=[((37.0, 31.0), 7.0)]
curroad=pos
#curroad=[[((35.0, 35.0), 0.0), ((45.0, 30.0), 17.0), ((57.0, 29.0), 18.0), ((63.0, 23.0), 2.0), ((65.0, 20.0), 6.0), ((65.0, 35.0), 3.0), ((64.0, 42.0), 9.0), ((56.0, 39.0), 36.0), ((56.0, 37.0), 6.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((47.0, 47.0), 13.0), ((55.0, 54.0), 26.0), ((57.0, 48.0), 23.0), ((55.0, 45.0), 13.0), ((53.0, 43.0), 14.0), ((50.0, 35.0), 19.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((41.0, 49.0), 10.0), ((40.0, 60.0), 21.0), ((45.0, 65.0), 9.0), ((49.0, 73.0), 25.0), ((57.0, 68.0), 15.0), ((55.0, 60.0), 16.0), ((53.0, 52.0), 11.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((26.0, 52.0), 9.0), ((24.0, 58.0), 19.0), ((30.0, 60.0), 16.0), ((27.0, 69.0), 10.0), ((31.0, 67.0), 3.0), ((35.0, 69.0), 23.0), ((37.0, 56.0), 5.0), ((37.0, 47.0), 6.0), ((35.0, 40.0), 16.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((41.0, 37.0), 16.0), ((49.0, 42.0), 13.0), ((61.0, 52.0), 3.0), ((65.0, 55.0), 14.0), ((63.0, 65.0), 8.0), ((62.0, 77.0), 20.0), ((49.0, 58.0), 10.0), ((31.0, 52.0), 27.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((27.0, 43.0), 9.0), ((15.0, 47.0), 16.0), ((13.0, 52.0), 36.0), ((10.0, 43.0), 9.0), ((6.0, 38.0), 16.0), ((14.0, 37.0), 11.0), ((20.0, 40.0), 12.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((25.0, 30.0), 3.0), ((12.0, 24.0), 13.0), ((10.0, 20.0), 19.0), ((4.0, 18.0), 35.0), ((5.0, 30.0), 2.0), ((11.0, 31.0), 7.0), ((17.0, 34.0), 3.0), ((26.0, 35.0), 15.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((26.0, 27.0), 27.0), ((19.0, 21.0), 10.0), ((15.0, 19.0), 1.0), ((11.0, 14.0), 18.0), ((18.0, 18.0), 17.0), ((20.0, 20.0), 8.0), ((22.0, 22.0), 2.0), ((25.0, 24.0), 20.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((49.0, 11.0), 18.0), ((45.0, 10.0), 18.0), ((42.0, 7.0), 5.0), ((30.0, 5.0), 8.0), ((23.0, 3.0), 7.0), ((5.0, 5.0), 16.0), ((15.0, 10.0), 20.0), ((24.0, 12.0), 5.0), ((32.0, 12.0), 7.0), ((35.0, 17.0), 7.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((36.0, 26.0), 18.0), ((45.0, 20.0), 11.0), ((47.0, 16.0), 25.0), ((46.0, 13.0), 8.0), ((44.0, 17.0), 9.0), ((40.0, 25.0), 9.0), ((37.0, 31.0), 14.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((55.0, 20.0), 19.0), ((60.0, 12.0), 31.0), ((67.0, 5.0), 25.0), ((55.0, 5.0), 29.0), ((53.0, 12.0), 6.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((20.0, 50.0), 5.0), ((15.0, 60.0), 17.0), ((20.0, 65.0), 12.0), ((15.0, 77.0), 9.0), ((6.0, 68.0), 30.0), ((2.0, 60.0), 5.0), ((8.0, 56.0), 27.0), ((2.0, 48.0), 1.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((15.0, 30.0), 26.0), ((16.0, 22.0), 41.0), ((18.0, 24.0), 22.0), ((20.0, 26.0), 9.0), ((22.0, 27.0), 11.0), ((35.0, 35.0), 0.0)]]

for i in range(13):
    if curroad==[]:
      break
    ind,flag,currentPath,ppos,curroad,splittpoint=tspPos(ind,flag,currentPath,ppos,curroad)
    print("ind:", ind)
    print("flag:", flag)
    print("需求拆分点:", ppos)
    print("currentPath:", currentPath)
    print("curroad:", curroad)
    print("splittpoint:", splittpoint)
    print("i=:",i)



#     ##
#     del reorder_p[index]
#     reorder_p=[p[0] for p in reorder_p]
#     neighbors,ind3 = find_k_nearest_neighbors(reorder_p, lst1[0],len(reorder_p))
#
#     print("离出发点最近neighbors:",neighbors)
#
#
#
# #画图
draw(path,locations,[splittpoint],pos[minpathindex])


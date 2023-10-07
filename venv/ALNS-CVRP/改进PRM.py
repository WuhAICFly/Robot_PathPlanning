# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:35:00 2023

@author: 86136
"""
from B_Spline import *
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera  # 保存动图时用，pip install celluloid
from queue import PriorityQueue
from math import sqrt
# parameter
N_SAMPLE = 500  # 采样点数目，即随机点集V的大小
N_KNN = 10  # 一个采样点的领域点个数
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True
class KKDTree:
    def __init__(self, points):
        def build_tree(points, depth):
            if not points:
                return None
            k = len(points[0])
            axis = depth % k
            points.sort(key=lambda x: x[axis])
            mid = len(points) // 2
            return {
                "point": points[mid],
                "left": build_tree(points[:mid], depth + 1),
                "right": build_tree(points[mid + 1:], depth + 1)
            }

        self.root = build_tree(points, depth=0)

    def search(self, query_point, k):
        def is_leaf(node):
            return node["left"] is None and node["right"] is None

        def distance(point1, point2):
            return sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]) ** 0.5

        def explore(node):
            if node is None:
                return
            priority = max(len(list(nearest_neighbors.queue)) - k + 1, 0) * distance(query_point, node["point"])
            if not is_leaf(node):
                axis = len(query_point) % len(node["point"])
                if query_point[axis] < node["point"][axis]:
                    explore(node["left"])
                    explore(node["right"])
                else:
                    explore(node["right"])
                    explore(node["left"])
            distance_to_point = distance(query_point, node["point"])
            if nearest_neighbors.qsize() < k or distance_to_point < -priority:
                nearest_neighbors.put((-distance_to_point, node["point"]))
                if nearest_neighbors.qsize() > k:
                    nearest_neighbors.get()

        nearest_neighbors = PriorityQueue()
        explore(self.root)
        return sorted([(-d, p) for d, p in nearest_neighbors.queue])
"""
kd-tree用于快速查找nearest-neighbor

query(self, x[, k, eps, p, distance_upper_bound]): 查询kd-tree附近的邻居


"""


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost  # 每条边权值
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y, merged_list , obstacle_x_list, obstacle_y_list, robot_radius,kdtree ,*, camara=None,
                 rng=None):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: 随机数构造器
    :return:
    """
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)
    # 采样点集生成
    start_time = time.time()
    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                       robot_radius,merged_list,
                                       obstacle_x_list, obstacle_y_list,
                                       obstacle_kd_tree,kdtree, rng)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("采样算法执行时间为：", elapsed_time, "秒")
    if show_animation:
        plt.plot(sample_x, sample_y, ".r")
    #plt.show()
    plt.savefig("result11.png")
    # 生成概率路图
    start_time = time.time()
    road_map = generate_road_map(sample_x, sample_y, robot_radius, merged_list, kdtree,obstacle_kd_tree)
    # 使用迪杰斯特拉规划路径
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("概率路图算法执行时间为：", elapsed_time, "秒")
    start_time = time.time()
    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y, camara)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("动态规划算法执行时间为：", elapsed_time, "秒")
    return rx, ry


def is_collision(sx, sy, gx, gy, rr,rrr,kdtree,obstacle_kd_tree):
    """判断是否发生碰撞,true碰撞，false不碰
        rr: 机器人半径
    """
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        if collision(x, y,rr,rrr,kdtree):
             return True
        # dist, _ = obstacle_kd_tree.query([x, y])  # 查询kd-tree附近的邻居
        # if dist <= rr:
        #     return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    # dist, _ = obstacle_kd_tree.query([gx, gy])
    # if dist <= rr:
    #     return True  # collision
    if collision(gx, gy, rr, rrr, kdtree):
        return True

    return False  # OK
#def is_collision():


def generate_road_map(sample_x, sample_y, rr , rrr, kdtree, obstacle_kd_tree):
    """
    概率路图生成

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """
    distss=[]
    indexess=[]
    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)
    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        # 对V中的每个点q，选择k个邻域点
        dists, indexes = sample_kd_tree.query([ix, iy], k=50)
        edge_id = []
        # for d, j in zip(dists, indexes):
        #  if d<=20:
        #   distss.append(d)
        #   indexess.append(j)
        #print("ix,iy:",[ix,iy])
        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]
            # 对每个领域点$q'$进行判断，如果$q$和$q'$尚未形成路径，则将其连接形成路径并进行碰撞检测，若无碰撞，则保留该路径。
            if not is_collision(ix, iy, nx, ny, rr,rrr,kdtree, obstacle_kd_tree):
            #if not collision(nx, ny,rr,rrr,kdtree):
              edge_id.append(indexes[ii])


            if len(edge_id) >= N_KNN:
                break
        road_map.append(edge_id)
        #print("road:",road_map)
        #plot_road_map(road_map, sample_x, sample_y)
        # time.sleep(2)
    #print("road_map:",road_map)
    return road_map

#@staticmethod
def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d
def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y, camara):
    """
    Dijkstra算法进行路径规划

    sx: 起始点x坐标 [m]
    sy: 起始点y坐标 [m]
    gx: 终点x坐标 [m]
    gy: 终点y坐标 [m]
    road_map: 构建好的道路地图 [m]
    sample_x: 采样点x坐标 [m]
    sample_y: 采样点y坐标 [m]
    camara: 可视化摄像机对象

    @return: 两个路径坐标列表 ([x1, x2, ...], [y1, y2, ...])，当无法找到路径时返回空列表
    """

    print("sample:",len(sample_x))
    write_to_file('output.txt', len(sample_x))
    # 起始节点
    start_node = Node(sx, sy, 0.0, -1)
    # 终止节点
    goal_node = Node(gx, gy, 0.0, -1)
    # 使用字典的方式构造开闭集合
    open_set, closed_set = dict(), dict()
    # 将起始节点加入开放集合中
    open_set[len(road_map) - 2] = start_node
    k = 0
    path_found = True
    # 开始迭代
    while True:
        # 如果开放集合为空
        if not open_set:
            print("Cannot find path")
            path_found = False
            break
        # 选择代价最小的节点
        c_id = min(open_set, key=lambda o:open_set[o].cost)
        current = open_set[c_id]
        # show graph(如果需要可视化)
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            k=k+1
            plt.pause(0.001)
            if camara != None:
                camara.snap()
        # 到达终点
        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break
        # 从开放集合中移除当前节点，加入到闭合集合中
        del open_set[c_id]
        closed_set[c_id] = current

        # 根据运动模型扩展搜索网格
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)
            # 如果节点已经在闭合集合中
            if n_id in closed_set:
                continue
            # 否则，如果节点已经在开放集合中
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            # 否则，将节点加入开放集合
            else:
                open_set[n_id] = node
    # 如果没有找到路径，返回空列表
    if path_found is False:
        return [], []
    # 生成最终路径
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index
    print("k:", k)
    return rx, ry

def rrdistance(pos1,pos2):
    new_pos = []
    common_index=[]
    for i, p2 in enumerate(pos2):
        for p1 in pos1:
            if p2[1][1] == p1[1][1]:
                new_points = [(p11, p22) for p11, p22 in zip(p1, p2)]
                new_pos.append(new_points)
                common_index.append(i)
                break
    return new_pos

def dashedCircle(x0,y0):
    # 圆心坐标和半径

    r = 5.5

    plt.scatter(x0, y0, marker='o',c='r', s=3)
    # 生成角度数组，并计算出圆周上每个点的坐标
    theta = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0

    # 绘制圆形
    fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal', adjustable='box')
    plt.plot(x, y,linestyle='--')

def collision(tx, ty,rr,rrr,kdtree):
    flag=0
    n=2
    nearest_k_points = []
    olddist=[]
    dist=[]
    oldindex=[]
    index=[]
    #nearest_neighbors = kdtree.search((tx, ty), k=10)
    dists, indexes =kdtree.query([tx, ty], k=n)#distance_upper_bound=5
    # indexes = kdtree.query_ball_point([tx, ty], r=5, p=2, return_sorted=True)
    # dists = np.linalg.norm(data[indices] - query_point, axis=1)
    #print("dists:",len(dists))
    for d,j in zip(dists, indexes):
        if(d<=5.5):
            olddist.append(d)
            oldindex.append(j)

    #print("olddist:",olddist)
    #print(oldindex)
    for i in range(len(oldindex)):
        index = oldindex[i]
        dist.append(olddist[i])
        nearest_k_points.append(kdtree.data[index])
    nearest_k_points = [(d[0], d[1]) for d in nearest_k_points]
    nearest_k_list = [(value, pos) for value, pos in zip(dist, nearest_k_points)]
    #new_list = [(x[0], x[1]) for x in nearest_k_points]
    #print("dist:",dist)
    #print("nearest_k_points:",nearest_k_list)
    newpoints = rrdistance(nearest_k_list, rrr)
    # print("rrr:",rrr)
    # print(kpos)
    # print(nearest_k_points)
    #print(newpoints)
    # print(newpoints[0][0])
    #print("邻居:", nearest_neighbors)
    pos = (tx, ty)
    targets = nearest_k_list
    for pos in newpoints:
        # drr=math.sqrt((pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2)
        #print(pos[0][0],(pos[0][1]+0.3))
        if pos[0][0] <= (pos[0][1]+0.3 ):
            flag = 1
            break
    # print("flag:",flag)
    return flag
def piont2line(x0,y0):
    point1 = (-5, 10)
    point2 = (55, 10)

    A = point2[1] - point1[1]
    B = point1[0] - point2[0]
    C = point2[0] * point1[1] - point1[0] * point2[1]
    d = abs(A * x0 + B * y0 + C) / math.sqrt(A ** 2 + B ** 2)

    #print("到直线的距离为：", d)
    return  d
def isPolygon(sample_x,sample_y):
    # 6 外圈障碍物
    ox=[]
    oy=[]
    ReadObstacle(6, ox, oy, 1.7, 8, 5)
    p1 = np.array([7.59297399782e-05, 5.93557160755e-07])
    p2 = np.array([-2.49960255623, -1.03229379654])
    points = np.linspace(p1, p2, num=20, endpoint=True)
    # 遍历每个点并将其x、y坐标加入列表中
    for point in points:
        ox.append(point[0] * 1.7 - 8)
        oy.append(point[1] * 1.7 + 5)
    for  x,y in zip(ox,oy):
        if x==sample_x and y==sample_y:
            return true
        else: return False
def Pos_probability(rr,rrr,kdtree,obstacle_kd_tree):
    distlist=[]
    xy=[]
    pbability=[]
    for x in range(-13,68):
        for y in range(-13,36):
            dist, index = obstacle_kd_tree.query([x, y])
            if (isInsidePolygon(x, y, polygon)): #and (not isPolygon(x,y)):
             if not collision(x,y,rr,rrr,kdtree):
               d0=piont2line(x,y)
               distlist.append(0.7*dist*dist+0.3*d0*d0)#
               xy.append((x,y))
    max_num = max(distlist)
    min_num = min(distlist)

    for d in distlist :
      pbability.append((max_num-d)/(max_num*len(distlist)-sum(distlist)))
    result = [math.exp(num * 10000) for num in pbability]
    total = sum(result)
    pbability = [round(num / total, 10) for num in result]
    sum_normalized_result = sum(pbability)
    print(max_num,sum_normalized_result)
    print(sum(distlist))
    print("distlist:",distlist)
    print("xy:",len(xy))
    print("pbability:",pbability)
    return xy,pbability
def sample_points(sx, sy, gx, gy, rr, rrr, ox, oy, obstacle_kd_tree, kdtree ,rng):
    """采样点集生成
    """
    # max_x = max(ox)
    # max_y = max(oy)
    # min_x = min(ox)
    # min_y = min(oy)
    max_x = 67.7
    max_y = 36
    min_x = -13
    min_y = -13
    print(max_x,min_x)
    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()
    xy,pbability=Pos_probability(rr,rrr,kdtree,kdtree)
    # 使用numpy.random.choice方法采样坐标点,采样50个点
    indices = np.random.choice(len(xy), 200, p=pbability)
    sampled_points = [xy[i] for i in indices]
    for x,y in sampled_points:
        sample_x.append(x)
        sample_y.append(y)
    print("count:",len(sampled_points))

    # while len(sample_x) <= 200:
    #     flag = 0
    #     #time.sleep(1)
    #     tx = (rng.random() * (max_x - min_x)) + min_x
    #     ty = (rng.random() * (max_y - min_y)) + min_y
    #     #print("tx,ty:",(tx,ty))
    #     # 在障碍物中查询离[tx, ty]最近的点的距离
    #     dist, index = obstacle_kd_tree.query([tx, ty])
    #     # flag=collision(tx, ty,rr, rrr,kdtree)
    #     # if flag==0:
    #     #  sample_x.append(tx)
    #     #  sample_y.append(ty)
    #      #print("sample:",(tx,ty))
    #     # 距离大于机器人半径，说明没有碰撞，将这个无碰撞的点加入V中，重复n次。
    #     if dist >= rr:
    #          sample_x.append(tx)
    #          sample_y.append(ty)
    #别忘了起点和目标点
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


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


def list2point(list1, list2):
    new_list = []
    #[[1,1],[2,2]]
    for x, y in zip(list1, list2):
        new_list.append([x, y])

    #new_list = list(zip(list1, list2)) ##[(1, 1), (2, 2)]点

    return new_list

    ## 递归的方式实现贝塞尔曲线
def bezier(Ps, n, t):
    """递归的方式求解贝塞尔点

    Args:
        Ps (_type_): 控制点，格式为numpy数组：array([[x1,y1],[x2,y2],...,[xn,yn]])
        n (_type_): n个控制点，即Ps的第一维度
        t (_type_): 时刻t

    Returns:
        _type_: 当前t时刻的贝塞尔点
    """
    if n == 1:
        return Ps[0]
    return (1 - t) * bezier(Ps[0:n - 1], n - 1, t) + t * bezier(Ps[1:n], n - 1, t)
#插入n点  points=[(1,2),(2,3)]
def insert_points(points, n):
    new_points = []
    new_points.append((points[0]))
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        for j in range(1, n + 1):
            x = x1 + (x2 - x1) * j / n
            y = y1 + (y2 - y1) * j / n
            new_points.append((x, y))
    return new_points


def Improve_Bezier(new_points, i):
    if (i == 0):
        group = new_points[0:10]
    else:
        group = new_points[i - 3:i + 10]

    # print(group)
    new_group = [[x, y] for x, y in group]
    return new_group
    print(new_group)

##计算夹角
def insert_22points(points):
    new_points = [points[0]]
    for i in range(1, len(points) - 1):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]
        angle = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angle = angle * 180 / math.pi
       # if angle > 90:
        print(i, angle)
    #         x1, y1 = p2
    #         x2, y2 = p3
    #         x, y = (x1 + x2) / 2, (y1 + y2) / 2
    #         new_points.append((x1, y1))
    #         new_points.append((x, y))
    #         new_points.append((x2, y2))
    #     else:
    #         new_points.append(p2)
    # new_points.append(points[-1])
    #return new_points
##计算夹角
def clcAngle(points):

   # 创建一个空列表用于存储连线夹角
  angles = []
  i=1
   # 遍历有序点
  for i in range(len(points) - 1):
    # 取出三个点的坐标
    x1, y1 = points[i-1]
    x2, y2 = points[i ]
    x3, y3 = points[i + 1]

    # 计算向量1
    vec1_x = x2 - x1
    vec1_y = y2 - y1

    # 计算向量2
    vec2_x = x3 - x2
    vec2_y = y3 - y2

    # 计算向量1和向量2的夹角，使用 atan2 函数
    angle = math.degrees(math.atan2(vec1_x * vec2_y - vec1_y * vec2_x, vec1_x * vec2_x + vec1_y * vec2_y))
    angles.append(angle)
  #for angle in angles:

  return angles

def clcComAngle(lst):
    lst_with_index = list(enumerate(lst))
    lst_with_index.sort(key=lambda x: x[1])
    print("从小到大排序：", lst_with_index)
    return  lst_with_index
def fisrtN(data,rx,ry,n):
    rrx = []
    rry = []
    # 取前N个元素
    top_n = data[:n]

    # 取出前两个元素的索引，并加1
    indexes = [x[0] + 1 for x in top_n]
    #print("indexs is:",indexes)
    # for i in indexes:
    rrx = [rx[i] for i in indexes]
    rry = [ry[i] for i in indexes]
    # print(rrx)
    # print(rry)
    rrx.insert(0, rx[0])
    rry.insert(0, ry[0])
    rrx.append(rx[len(rx)-1])
    rry.append(ry[len(ry)-1])
    return rrx, rry

def circle(x1, y1, x2, y2, x3, y3):
# 将输入的字符串转换为浮点数类型
 #x1, y1, x2, y2, x3, y3 = 0,1,2,3,6,8

# 计算圆心坐标和半径
# 求出两条中垂线的交点(x0,y0)
 k1 = (y2 - y1) / (x2 - x1)
 k2 = (y3 - y2) / (x3 - x2)
 x0 = (k1 * k2 * (y3 - y1) + k1 * (x2 + x3) - k2 * (x1 + x2)) / (2 * (k1 - k2))
 y0 = -1 * (x0 - (x1 + x2) / 2) / k1 + (y1 + y2) / 2
 r = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

# 输出圆心坐标和半径
# print("圆心坐标为：({:.2f},{:.2f})".format(x0, y0))
# print("半径为：{:.2f}".format(r))

# 画出圆
 theta = [i * (2 * math.pi) / 1000 for i in range(1001)]
 x = [r * math.cos(t) + x0 for t in theta]
 y = [r * math.sin(t) + y0 for t in theta]

 plt.plot(x, y,c='y')
 plt.axis('equal')
 return r, (x0,y0)
#plt.show()

def clccircle(Cpionts):
    # 做障碍物外接圆，并求半径r和圆心
    R = []
    cPos = []
    for ppos in Cpionts:
        r, cpos = circle(ppos[0], ppos[1], ppos[2], ppos[3], ppos[4], ppos[5])
        R.append(r)
        cPos.append(cpos)
        plt.scatter(cpos[0], cpos[1], c='r', s=1)
    return R,cPos
   # print("半径R:",R)
   # print("圆心:",cPos)
def b_splin(p):
    k = 3  # k阶、k-1次B样条



    # 控制点
    P = p

    n = len(P) - 1  # 控制点个数-1

    ## 生成B样条曲线

    path = []  # 路径点数据存储
    Bik_u = np.zeros((n + 1, 1))
    NodeVector = U_quasi_uniform(n, k)
    for u in np.arange(0, 1, 0.005):
        for i in range(n + 1):
            Bik_u[i, 0] = BaseFunction(i, k, u, NodeVector)
        p_u = P.T @ Bik_u
        path.append(p_u)
    path = np.array(path)
    ## 画图
    fig = plt.figure(1)
    # plt.ylim(-4, 4)
    # plt.axis([-10, 100, -15, 15])
    camera = Camera(fig)

    for i in range(len(path)):
        # plt.cla()

        plt.plot(P[:, 0], P[:, 1], 'ro')
        plt.plot(P[:, 0], P[:, 1], 'y')
        # 设置坐标轴显示范围
        # plt.axis('equal')
        plt.gca().set_aspect('equal')
        # 绘制路径

        plt.plot(path[0:i, 0], path[0:i, 1], 'g')  # 路径点
        # plt.pause(0.001)
    #     camera.snap()
    # animation = camera.animate()
    # animation.save('trajectory.gif')
    #plt.show()
def isInsidePolygon(x, y, polygon):
    polygon = [(-11.91, 15.08), (-12.09, 3.32), (-7.72, 5.01), (-2.56, -7.11), (5.73, -12.72), (8.22, -12.18),
               (13.93, -7.46), (32.46, -13.43), (40.65, -10.40),(45.37, -2.74), (53.84, -3.19), (63.55, 10.62),
               (67.02, 13.83), (57.67, 23.90), (52.86, 21.58),(40.12, 36.01), (27.11, 27.19), (19.63, 27.26),
                (3.5, 22.81), (1.1, 19.4)]
    n = len(polygon)
    if n < 3:
        return False
    crossings = 0
    for i in range(n):
        j = (i + 1) % n
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1])):
            crossings += 1
    return crossings % 2 == 1


def polygon():
    pos=[(-11.91,15.08),(-12.09,3.32),(-7.72,5.01),(-2.56,-7.11),(5.73,-12.72),(8.22,-12.18),(13.93,-7.46),(32.46,-13.43),(40.65,-10.40),
         (45.37,-2.74),(53.84,-3.19),(63.55,10.62),(67.02,13.83),(57.67,23.90),(52.86,21.58),(40.12,36.01),(27.11,27.19),(15.53,28.17),
         (6.62,26.12),(1.99,23.27),(-0.60,19.09),(-11.91,15.08)]
    x, y = zip(*pos)
    fig = plt.figure(1)
    plt.plot(x, y,color='y', linewidth=4)
    #plt.show()

def write_to_file(file_name, data):
        with open(file_name, 'a') as f:
            data = str(data)
            f.write(data)
            f.write(' ')
            # f.write('\n')

def main(rng=None):
    print(" start!!")
    fig = plt.figure(1)

    # camara = Camera(fig)  # 保存动图时使用
    camara = None
    # start and goal position
    sx = -5.0  # [m]
    sy = 10.0  # [m]
    gx = 54.0  # [m]
    gy = 10.0  # [m]
    robot_size = 1.3  # [m]

    ox = []
    oy = []
    x = []
    y = []
    oxx = []
    oyy = []
    rrr = []
    #6 外圈障碍物
    ReadObstacle(6, ox, oy, 1.7, 8, 5)
    p1 = np.array([7.59297399782e-05, 5.93557160755e-07])
    p2 = np.array([-2.49960255623, -1.03229379654])
    points = np.linspace(p1, p2, num=20, endpoint=True)
    # 遍历每个点并将其x、y坐标加入列表中
    for point in points:
        ox.append(point[0] * 1.7 - 8)
        oy.append(point[1] * 1.7 + 5)
    polygon()
    # 1-4障碍物
    for i in range(6):
        ReadObstacle(i, ox, oy, 1, 0, 0)
    # 7障碍物
    ReadObstacle(7, ox, oy, 1.2, -20, 0)
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")

        plt.axis("equal")
        if camara != None:
            camara.snap()
    #plt.grid(None)
    # plt.show()
    #重构障碍物，求圆心和r
    Cpionts = [[2.79, -4.08, 4.57, -5.06, 4.66, -2.56], [25.06, 0.91, 28.09, 4.74, 33.44, 0.29],
               [19.00, 11.33, 16.95, 7.95, 22.57, 7.86], [17.31, 1.27, 18.82, -1.67, 20.16, 1.00],
               [6.00, -8.53, 8.67, -8.89, 8.13, -6.66],
               [11.52, 19.44, 7.69, 18.37, 7.78, 23.18], [11.61, 15.25, 11.43, 19.53, 14.28, 17.30],
               [11.61, 15.25, 7.24, 14.36, 10.36, 12.40], [1.8, 12.3, 6.98, 14.01, 3.8, 9.2],
               [5.37, 7.86, 8.67, 5.19, 10.09, 9.46],
               [13.75, 5.10, 9.65, 1.71, 9.47, 5.72], [9.65, -1.14, 5.55, 5.01, 4.84, -1.14],
               [7.69, 12.94, 10.01, 9.64, 10.36, 12.23], [7.69, 18.19, 8.05, 16.50, 9.74, 18.02],
               [41.99, 13.12, 40.65, 9.11, 45.73, 13.03]]
    rr, ccpos = clccircle(Cpionts)
    print(rr)
    print(ccpos)
    for pos in ccpos:
        #rrr.append(pos[0][0])
        oxx.append(pos[0])
        oyy.append(pos[1])

        # # 6 外圈障碍物
        # ReadObstacle(6, ox, oy, 1.7, 8, 5)
        # p1 = np.array([7.59297399782e-05, 5.93557160755e-07])
        # p2 = np.array([-2.49960255623, -1.03229379654])
        # points = np.linspace(p1, p2, num=20, endpoint=True)
        # # 遍历每个点并将其x、y坐标加入列表中
        # for point in points:
        #     rr.append(0)
        #     oxx.append(point[0] * 1.7 - 8)
        #     oyy.append(point[1] * 1.7 + 5)

    merged_list = [(value, pos) for value, pos in zip(rr, ccpos)]
    #print(merged_list)
    obstacle_kd_tree = KDTree(np.vstack((oxx, oyy)).T)
    kdtree = KDTree(np.vstack((oxx, oyy)).T)#KKDTree(ccpos)
    #plt.title('w1=' )
    plt.xlabel("X/m")
    plt.ylabel("Y/m")
    start_time = time.time()

    rx,ry = prm_planning(sx, sy, gx, gy, merged_list, ox, oy, robot_size, kdtree, camara=camara, rng=rng)
    if len(rx)==0:
        with open('output.txt', 'a') as f:
            data = str(0)
            f.write(data)
            f.write('\n')
    else:
      write_to_file('output.txt', 1)
    assert rx, 'Cannot found path'
    end_time = time.time()
    elapsed_time = end_time - start_time
    write_to_file('output.txt', elapsed_time)
    print("该算法执行时间为：", elapsed_time, "秒")

    #if show_animation:
    print(rx)
    print(ry)
    plt.plot(rx, ry, "-g")
    #plt.show()
    #plt.scatter(rx, ry, c='y', s=5)
    plt.grid(False)


    #plt.show()
    #print(rx)
    # for i in range(len(rx)):
    #     rx[i] = round(rx[i], 2)
    # for i in range(len(ry)):
    #     ry[i] = round(ry[i], 2)
    #Ps = np.stack((rx, ry), axis=1)
    rx = list(reversed(rx))#逆序，从起点开始
    ry = list(reversed(ry))
    ppos=[(x,y)for(x,y)in zip(rx,ry)]
    total_distance = 0
    for i in range(1, len(ppos)):
        total_distance += math.sqrt((ppos[i][0] - ppos[i - 1][0]) ** 2 + (ppos[i][1] - ppos[i - 1][1]) ** 2)

    print("路径长度为：", total_distance)
    with open('output.txt', 'a') as f:
        data = str(total_distance)
        f.write(data)
        f.write('\n')
    print("xy:",ppos)
    print(rx)
    print(ry)
    new_list = list(zip(rx, ry))
    x_ = []
    y_ = []
    Ps = np.array(list2point(rx, ry))
    num=len(Ps)
    print(num)
    #b_splin(Ps)
    # for t in np.arange(0, 1, 0.01):
    #     pos = bezier(Ps, len(Ps), t)
    #     x_.append(pos[0])
    #     y_.append(pos[1])
    # plt.scatter(x_, y_, c='r', s=1)
    # angles = clcAngle(new_list)#计算角度
    # print(angles)
    # #求补角
    # new_angles=[]
    # for num in angles:
    #     new_angles.append(180 - abs(num))
    # print(new_angles)
    #
    #  #比较角度排序
    # data=clcComAngle(new_angles)
    #
    # print(len(rx))
    # #从角度小到大找前N点
    # rrx,rry=fisrtN(data, rx, ry, 10)
    # new_list1 = list(zip(rrx, rry))
    # print(new_list1)
    # # x_ = []
    # # y_ = []
    # # Ps = np.array(list2point(rrx, rry))
    # # for t in np.arange(0, 1, 0.01):
    # #     pos = bezier(Ps, len(Ps), t)
    # #     x_.append(pos[0])
    # #     y_.append(pos[1])
    # # plt.scatter(x_, y_, c='r', s=1)
    # rx=[[-5.0,-3.3535719964920805,-1.8597723300270186],[-1.8597723300270186,-0.12685039058207437,13.453710692904487],[13.453710692904487,15.840972284822897,25.0]]
    # ry=[[10.0,10.190506040070954,-1.269685746547557],[-1.269685746547557,-6.532514133498683,-0.9629071348527614],[-0.9629071348527614,2.800847757856886,5.0]]
    #
    # print(new_list)
    # #贝塞尔曲线
    # x_ = []
    # y_ = []
    # for rxx, ryy in zip(rx, ry):
    #     Ps = np.array(list2point(rxx, ryy))
    #     for t in np.arange(0, 1, 0.01):
    #       pos = bezier(Ps, len(Ps), t)
    #       x_.append(pos[0])
    #       y_.append(pos[1])
    #     #plt.plot(Ps[:, 0], Ps[:, 1])
    # plt.scatter(x_, y_, c='r', s=1)
    # plt.savefig("bezier.png")
    plt.show()

if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:35:00 2023

@author: 86136
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera  # 保存动图时用，pip install celluloid

# parameter
N_SAMPLE = 500  # 采样点数目，即随机点集V的大小
N_KNN = 10  # 一个采样点的领域点个数
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True

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


def prm_planning(start_x, start_y, goal_x, goal_y, obstacle_x_list, obstacle_y_list, robot_radius, *, camara=None,
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
    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                       robot_radius,
                                       obstacle_x_list, obstacle_y_list,
                                       obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")
    #plt.show()
    # 生成概率路图
    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)
    # 使用迪杰斯特拉规划路径

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y, camara)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
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
        dist, _ = obstacle_kd_tree.query([x, y])  # 查询kd-tree附近的邻居
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    概率路图生成

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        # 对V中的每个点q，选择k个邻域点
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]
            # 对每个领域点$q'$进行判断，如果$q$和$q'$尚未形成路径，则将其连接形成路径并进行碰撞检测，若无碰撞，则保留该路径。
            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    # plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y, camara):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: 构建好的路图 [m]
    sample_x: 采样点集x [m]
    sample_y: 采样点集y [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)
    # 使用字典的方式构造开闭集合
    # openList表由待考察的节点组成， closeList表由已经考察过的节点组成。
    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True
    # 步骤与A星算法一致
    while True:
        # 如果open_set是空的
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)
            if camara != None:
                camara.snap()

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    """采样点集生成
    """
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        # 在障碍物中查询离[tx, ty]最近的点的距离
        dist, index = obstacle_kd_tree.query([tx, ty])

        # 距离大于机器人半径，说明没有碰撞，将这个无碰撞的点加入V中，重复n次。
        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
    # 别忘了起点和目标点
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
def ReadObstacle(i,list2,list3,times,dtax,dtay):
    filename = str(i)+"." + "txt"
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
            list2.append(float(data[1])*times-dtax)
            list3.append(float(data[2])*times+dtay)

def list2point(list1,list2):
    new_list = []

    for x, y in zip(list1, list2):
        new_list.append([x, y])

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

def plot_smoothed_path(smoothed_path):
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], "-b")
    plt.show()

def insert_points(points,n):
    new_points = []
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        for j in range(1, n+1):
            x = x1 + (x2-x1) * j / n
            y = y1 + (y2-y1) * j / n
            new_points.append((x, y))
    return new_points

def Improve_Bezier(new_points,i):


        if (i == 0):
            group = new_points[0:10]
        else:
            group = new_points[i - 3:i + 10]

        # print(group)
        new_group = [[x, y] for x, y in group]
        return  new_group
        print(new_group)

def insert_22points(points):
    new_points = [points[0]]
    for i in range(1, len(points)-1):
        p1 = points[i-1]
        p2 = points[i]
        p3 = points[i+1]
        angle = math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        angle = angle * 180 / math.pi
        if angle > 90:
            print(i,angle)
            x1, y1 = p2
            x2, y2 = p3
            x, y = (x1+x2)/2, (y1+y2)/2
            new_points.append((x1, y1))
            new_points.append((x, y))
            new_points.append((x2, y2))
        else:
            new_points.append(p2)
    new_points.append(points[-1])
    return new_points

def main(rng=None):
    print(" start!!")
    fig = plt.figure(1)

    # camara = Camera(fig)  # 保存动图时使用
    camara = None
    # start and goal position
    sx = -5.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 10.0  # [m]
    robot_size = 1.3  # [m]

    ox = []
    oy = []
    x = []
    y = []

    #6 外圈障碍物
    ReadObstacle(6, ox, oy, 1.7, 8, 5)
    p1 = np.array([7.59297399782e-05, 5.93557160755e-07])
    p2 = np.array([-2.49960255623, -1.03229379654])
    points = np.linspace(p1, p2, num=20, endpoint=True)
    # 遍历每个点并将其x、y坐标加入列表中
    for point in points:
        ox.append(point[0]*1.7-8)
        oy.append(point[1]*1.7+5)
    #1-4障碍物
    for i in range(6):

      ReadObstacle(i, ox, oy, 1, 0,0)
    #7障碍物
    ReadObstacle(7, ox, oy, 1.2, -20, 0)
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(None)
        plt.axis("equal")
        if camara != None:
            camara.snap()
    plt.grid(None)
    #plt.show()
    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, camara=camara, rng=rng)
    assert rx, 'Cannot found path'
    

    plt.show()
if __name__ == '__main__':
    main()


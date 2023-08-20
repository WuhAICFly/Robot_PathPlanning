# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:35:00 2023

@author: 86136
"""
import time
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
    start_time = time.time()
    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)
    # 使用迪杰斯特拉规划路径
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("概率图生成算法执行时间为：", elapsed_time, "秒")
    start_time = time.time()
    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y, camara)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("动态规划算法执行时间为：", elapsed_time, "秒")
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
    print("road_map:",len(road_map))
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
    print("sample:", len(sample_x))
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
    # max_x = 54
    # max_y = 23
    # min_x = -13
    # min_y = -9

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
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")

        plt.axis("equal")
        if camara != None:
            camara.snap()
    plt.grid(False)
    #plt.show()
    start_time = time.time()
    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, camara=camara, rng=rng)
    assert rx, 'Cannot found path'
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("该算法执行时间为：", elapsed_time, "秒")
    #if show_animation:

    plt.plot(rx, ry, "-g")
    #plt.show()
    plt.scatter(rx, ry, c='y', s=5)
    plt.grid(None)
    plt.savefig("result.png")

    plt.show()
    #print(rx)
    # for i in range(len(rx)):
    #     rx[i] = round(rx[i], 2)
    # for i in range(len(ry)):
    #     ry[i] = round(ry[i], 2)
    #Ps = np.stack((rx, ry), axis=1)
    rx = list(reversed(rx))#逆序，从起点开始
    ry = list(reversed(ry))
    print(rx)
    print(ry)
    new_list = list(zip(rx, ry))
    angles = clcAngle(new_list)#计算角度
    print(angles)
    #求补角
    new_angles=[]
    for num in angles:
        new_angles.append(180 - abs(num))
    print(new_angles)

     #比较角度排序
    data=clcComAngle(new_angles)

    print(len(rx))
    #从角度小到大找前N点
    rrx,rry=fisrtN(data, rx, ry, 10)
    new_list1 = list(zip(rrx, rry))
    print(new_list1)
    # x_ = []
    # y_ = []
    # Ps = np.array(list2point(rrx, rry))
    # for t in np.arange(0, 1, 0.01):
    #     pos = bezier(Ps, len(Ps), t)
    #     x_.append(pos[0])
    #     y_.append(pos[1])
    # plt.scatter(x_, y_, c='r', s=1)
    rx=[[-5.0,-3.3535719964920805,-1.8597723300270186],[-1.8597723300270186,-0.12685039058207437,13.453710692904487],[13.453710692904487,15.840972284822897,25.0]]
    ry=[[10.0,10.190506040070954,-1.269685746547557],[-1.269685746547557,-6.532514133498683,-0.9629071348527614],[-0.9629071348527614,2.800847757856886,5.0]]

    print(new_list)
    #贝塞尔曲线
    x_ = []
    y_ = []
    for rxx, ryy in zip(rx, ry):
        Ps = np.array(list2point(rxx, ryy))
        for t in np.arange(0, 1, 0.01):
          pos = bezier(Ps, len(Ps), t)
          x_.append(pos[0])
          y_.append(pos[1])
        #plt.plot(Ps[:, 0], Ps[:, 1])
    #plt.scatter(x_, y_, c='r', s=1)
    plt.savefig("bezier.png")
    plt.show()

if __name__ == '__main__':
    main()


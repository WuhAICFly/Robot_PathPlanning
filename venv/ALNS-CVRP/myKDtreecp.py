import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from queue import PriorityQueue
from math import sqrt
show_animation=True
N_SAMPLE=500



nearest_neighbors = PriorityQueue()


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def distance(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)



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

def ObstaclePoints(sx,sy,gx,gy):
    ox = []
    oy = []
    camara=None
    # camara = Camera(fig)  # 保存动图时使用
    # 6 外圈障碍物
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
    return ox,oy



def is_collision(x, y, rr, obstacle_kd_tree):
    """判断是否发生碰撞,true碰撞，false不碰
        rr: 机器人半径
    """


    dist, _ = obstacle_kd_tree.query([x, y])  # 查询kd-tree附近的邻居
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
    edge_ix = []
    edge_iy = []
    indexs =[]
    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):


            #进行碰撞检测，若碰撞，则保留该点。
            if  is_collision(ix, iy, rr, obstacle_kd_tree):
                edge_ix.append(ix)
                edge_iy.append(iy)
                indexs.append(i)
    print(edge_ix)
    print(edge_iy)
    print(indexs)
    plt.plot(edge_ix,edge_iy, "y")
    return edge_ix,edge_iy,indexs
#求bezier碰撞分段x,y
def collisionPosSegment(indexs,rx,ry):
    # 按照索引段划分rx和ry
    rx_segments = []
    ry_segments = []
    index_segments = fix_index(indexs)
    for segment in index_segments:
        rx_segments.append(rx[segment[0] :segment[-1]+1])
        ry_segments.append(ry[segment[0] :segment[-1]+1])

       # print(rx_segments)
       # print(ry_segments)
    return rx_segments,ry_segments,index_segments
#求bezier碰撞段界限
def limit(my_list1):

    new_list = []
    seen = set()

    for sublist in my_list1:
        if len(sublist) == 1:
            element = sublist[0]
            if element not in seen:
                new_list.append(element)
                seen.add(element)
        else:
            start, end = sublist[0], sublist[-1]
            if start not in seen:
                new_list.append(start)
                seen.add(start)
            if end not in seen:
                new_list.append(end)
                seen.add(end)
    print(sorted(new_list))
    return new_list
#按bezier 碰撞点查找对应的PRM的x
def mmpPRM(list,my_list2):
    sublists = []
    for i in range(len(list)):
        if i == 0:
            sublists.append([x for x in my_list2 if x <= list[i]])
        elif i == len(list) - 1:
            sublists.append([x for x in my_list2 if x > list[i - 1]])
        else:
            sublists.append([x for x in my_list2 if x <= list[i] and x > list[i - 1]])

    result = []
    for sublist in sublists:
        if not sublist:
            result.append([])
        else:
            result.append(sublist)
    print(result)
    return result

# list 子列表首插入前一个子列表尾元素
def frist_append(old_list):


    result = [old_list[0]]

    for i in range(1, len(old_list)):
        prev = result[-1][-1]
        old_list[i].insert(0, prev)
        result.append(old_list[i])
    print(result)
    return result
#求x对应的y
def x2y(list1,list2):
 res = []
 result=[]
 for x,y in list2:
    for sublist in list1:
        if x in sublist:
            res.append(y)
            break
    else:
        continue
 list2 = res

 for sublist in list1:
    length = len(sublist)
    result.append(list2[:length])
    list2 = list2[length:]
 print(result)
 return result

#遍历索引号，如果断续则分成一段
def fix_index(index_list):
    index_segments = []  # 存储所有的索引段

    current_segment = [index_list[0]]  # 当前的索引段，初始值为第一个索引号

    for i in range(1, len(index_list)):  # 从第二个索引号开始遍历
        if index_list[i] == index_list[i - 1] + 1:  # 如果当前索引号与前一个索引号连续
            current_segment.append(index_list[i])  # 将当前索引号加入当前的索引段
        else:  # 如果当前索引号与前一个索引号不连续
            index_segments.append(current_segment)  # 将当前索引段加入索引段列表
            current_segment = [index_list[i]]  # 开始一个新的索引段

    # 处理最后一个索引段
    index_segments.append(current_segment)
    print(index_segments)
    return index_segments
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

def list2point(list1, list2):
    new_list = []
    #[[1,1],[2,2]]
    for x, y in zip(list1, list2):
        new_list.append([x, y])

    #new_list = list(zip(list1, list2)) ##[(1, 1), (2, 2)]点

    return new_list

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
    return R,cPos
   # print("半径R:",R)
   # print("圆心:",cPos)
#def collision():


def main(rng=None):
    print(" start!!")
    fig = plt.figure(1)
    # start and goal position
    sx = -5.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 10.0  # [m]
    x=[]
    y=[]
    oxx=[]
    oyy=[]
    robot_size = 1.3  # [m]
    #障碍物坐标，构建KDtree
    ox,oy = ObstaclePoints(sx,sy,gx,gy)
    #obstacle_kd_tree = KDTree(np.vstack((ox,oy)).T)


    #上边界路线
    #rx=[-5.0, -2.349289136285435, -1.0858752184665565, 4.63812992990092, 5.3423344023570145, 15.766350619929433,18.407123230118838, 21.18510023012506, 26.082982517220437, 25.0]
    #ry=[10.0, 12.40756272425698, 15.583633410112622, 15.072065169356922, 23.453619715407545, 26.74849314301486, 18.42202243050772, 13.064246446844368, 9.764583271620218, 5.0]
    #中路路线
    #rx=[-5.0, -3.0896683080131737, -2.0375382658638337, -0.27426048691631166, 0.8730135313779908, 0.7034404293426775,2.9519279084765913, 13.982652667934463, 15.005987660437533, 18.38589197455267, 21.244933095435222,22.3783720690584, 25.0]
    #ry=[10.0, 8.192206949321864, 7.212839271716284, 3.8384268254357536, 1.3976093034049253, -5.135953318920873, -6.578341220459899, -4.069077318655813, -3.9328493566595224, -4.320023766412092, -4.46608613981808, -0.9887107645069726, 5.0]
    Cpionts = [[2.79, -4.08, 4.57, -5.06, 4.66, -2.56], [25.06, 0.91, 28.09, 4.74, 33.44, 0.29],[19.00, 11.33, 16.95, 7.95, 22.57, 7.86], [17.31, 1.27, 18.82, -1.67, 20.16, 1.00],[6.00, -8.53, 8.67, -8.89, 8.13, -6.66],
               [11.52, 19.44, 7.69, 18.37, 7.78, 23.18], [11.61, 15.25, 11.43, 19.53, 14.28, 17.30],[11.61, 15.25, 7.24, 14.36, 10.36, 12.40], [1.8, 12.3, 6.98, 14.01, 3.8, 9.2],[5.37, 7.86, 8.67, 5.19, 10.09, 9.46],
               [13.75, 5.10, 9.65, 1.71, 9.47, 5.72], [9.65, -1.14, 5.55, 5.01, 4.84, -1.14],[7.69, 12.94, 10.01, 9.64, 10.36, 12.23], [7.69, 18.19, 8.05, 16.50, 9.74, 18.02],[41.99,13.12,40.65,9.11,45.73,13.03]]
    rr,ccpos=clccircle(Cpionts)
    for pos in ccpos:
        oxx.append(pos[0])
        oyy.append(pos[1])
    obstacle_kd_tree = KDTree(np.vstack((oxx,oyy)).T)
    kdtree = KKDTree(ccpos)
    nearest_neighbors = kdtree.search((5, 4), k=5)
    # for distance, point in nearest_neighbors:
    #     p1 = Point(1, 2)
    #     p2 = Point(2, 3)
    #
    #     print(p1.distance(p2))
    #
    #     print("point:", point, "distance:", sqrt(distance))
    rx, ry = prm_planning(sx, sy, gx, gy, oxx, oyy, robot_size, camara=camara, rng=rng)
    assert rx, 'Cannot found path'

    # if show_animation:

    plt.plot(rx, ry, "-g")
    # plt.show()
    plt.scatter(rx, ry, c='y', s=5)
    plt.grid(None)
    plt.savefig("result.png")
    # points = list(map(list, zip(rx, ry)))
    # #原始曲线
    # plt.plot(rx, ry, "-g")
    # #做全局bezier曲线拟合
    # Ps =np.array(list2point(rx, ry))
    # for t in np.arange(0, 1, 0.01):
    #     pos = bezier(Ps, len(Ps), t)
    #     x.append(pos[0])
    #     y.append(pos[1])
    # plt.scatter(x, y, c='b', s=1)
    # # plt.savefig("bezier.png")
    # # 生成概率路图
    # ix,iy,indexs = generate_road_map(x, y, robot_size, obstacle_kd_tree)
    # #bezier碰撞部分连续根据断续分段
    # rx_segments, ry_segments, index_segments= collisionPosSegment(indexs,x,y)
    # limits=limit(rx_segments)
    # old_x=mmpPRM(limits,rx)
    # old_y=x2y(old_x, points)
    # rrx=frist_append(old_x)
    # rry = frist_append(old_y)
    # ##
    # #rrx=[[-5.0, -3.0896683080131737, -2.0375382658638337, -0.27426048691631166, 0.8730135313779908, 0.7034404293426775, 2.9519279084765913], [0.8730135313779908, 0.7034404293426775,2.9519279084765913, 13.982652667934463, 15.005987660437533, 18.38589197455267, 21.244933095435222, 22.3783720690584, 25.0]]
    # #rry=[[10.0, 8.192206949321864, 7.212839271716284, 3.8384268254357536, 1.3976093034049253, -5.135953318920873, -6.578341220459899], [1.3976093034049253, -5.135953318920873,-6.578341220459899, -4.069077318655813, -3.9328493566595224, -4.320023766412092, -4.46608613981808, -0.9887107645069726, 5.0]]
    # xx=[]
    # yy=[]
    # for rrxx, rryy in zip(rrx, rry):
    #  if len(rrxx)>0 and len(rryy)>0:
    #
    #
    #    Ps = np.array(list2point(rrxx, rryy))
    #    for t in np.arange(0, 1, 0.01):
    #     pos = bezier(Ps, len(Ps), t)
    #     xx.append(pos[0])
    #     yy.append(pos[1])
    #    plt.scatter(xx, yy, c='r', s=1)
    #  else:
    #      continue

    plt.grid(None)
    plt.show()



if __name__ == '__main__':
    main()
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

show_animation=True
N_SAMPLE=500
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
    robot_size = 1.3  # [m]
    ox,oy = ObstaclePoints(sx,sy,gx,gy)
    obstacle_kd_tree = KDTree(np.vstack((ox,oy)).T)

    #上边界路线
    rx=[-5.0, -2.349289136285435, -1.0858752184665565, 4.63812992990092, 5.3423344023570145, 15.766350619929433,18.407123230118838, 21.18510023012506, 26.082982517220437, 25.0]
    ry=[10.0, 12.40756272425698, 15.583633410112622, 15.072065169356922, 23.453619715407545, 26.74849314301486, 18.42202243050772, 13.064246446844368, 9.764583271620218, 5.0]
    #中路路线
    rx=[-5.0, -3.0896683080131737, -2.0375382658638337, -0.27426048691631166, 0.8730135313779908, 0.7034404293426775,2.9519279084765913, 13.982652667934463, 15.005987660437533, 18.38589197455267, 21.244933095435222,22.3783720690584, 25.0]
    ry=[10.0, 8.192206949321864, 7.212839271716284, 3.8384268254357536, 1.3976093034049253, -5.135953318920873, -6.578341220459899, -4.069077318655813, -3.9328493566595224, -4.320023766412092, -4.46608613981808, -0.9887107645069726, 5.0]
    points = list(map(list, zip(rx, ry)))
    #print(points)
    plt.plot(rx, ry, "-g")
    #plt.scatter(rx, ry, c='y', s=5)
    # plt.show()
    Ps =np.array(list2point(rx, ry))
    for t in np.arange(0, 1, 0.01):
        pos = bezier(Ps, len(Ps), t)
        x.append(pos[0])
        y.append(pos[1])
    # print(x)
    # print(y)

    plt.scatter(x, y, c='b', s=1)
    # plt.savefig("bezier.png")
    # 生成概率路图
    ix,iy,indexs = generate_road_map(x, y, robot_size, obstacle_kd_tree)
    #bezier碰撞部分连续根据断续分段
    rx_segments, ry_segments, index_segments= collisionPosSegment(indexs,x,y)
    limits=limit(rx_segments)
    old_x=mmpPRM(limits,rx)
    old_y=x2y(old_x, points)
    rrx=frist_append(old_x)
    rry = frist_append(old_y)
    ##
    #rrx=[[-5.0, -3.0896683080131737, -2.0375382658638337, -0.27426048691631166, 0.8730135313779908, 0.7034404293426775, 2.9519279084765913], [0.8730135313779908, 0.7034404293426775,2.9519279084765913, 13.982652667934463, 15.005987660437533, 18.38589197455267, 21.244933095435222, 22.3783720690584, 25.0]]
    #rry=[[10.0, 8.192206949321864, 7.212839271716284, 3.8384268254357536, 1.3976093034049253, -5.135953318920873, -6.578341220459899], [1.3976093034049253, -5.135953318920873,-6.578341220459899, -4.069077318655813, -3.9328493566595224, -4.320023766412092, -4.46608613981808, -0.9887107645069726, 5.0]]
    xx=[]
    yy=[]
    for rrxx, rryy in zip(rrx, rry):
     if len(rrxx)>0 and len(rryy)>0:


       Ps = np.array(list2point(rrxx, rryy))
       for t in np.arange(0, 1, 0.01):
        pos = bezier(Ps, len(Ps), t)
        xx.append(pos[0])
        yy.append(pos[1])
       plt.scatter(xx, yy, c='r', s=1)
     else:
         continue

    plt.grid(None)
    plt.show()



if __name__ == '__main__':
    main()
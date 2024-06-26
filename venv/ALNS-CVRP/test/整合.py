# import math
#
# # 给定列表
# lst = [((45.0, 30.0), 17.0), ((57.0, 29.0), 18.0), ((63.0, 23.0), 2.0)]
#
# # 初始化距离之和为0
# total_distance = 0
#
# # 遍历列表中的每个点
# for point, _ in lst:
#     # 计算点到原点的距离
#     distance = math.sqrt((point[0] - 0)**2 + (point[1] - 0)**2)
#     # 将距离加入总距离
#     total_distance += distance
#
# # 打印总距离
# print(total_distance)
from scipy.spatial import KDTree
from typing import List, Tuple
import numpy as np


def find_k_nearest_neighbors(points: List[Tuple[float, float]], target_point: Tuple[float, float], k: int) -> List[
    Tuple[Tuple[float, float], float]]:
    # 构建kdtree
    tree = KDTree(points)

    # 找k个最近邻
    dist, ind = tree.query([target_point], k=k)

    # 返回k个最近邻的坐标和距离
    return [(tuple(tree.data[i]), d) for i, d in zip(ind[0], dist[0])]
# 定义点列表
points = [(41, 49), (35, 17), (55, 45), (55, 20), (15, 30), (25, 30), (20, 50), (10, 43), (55, 60), (30, 60), (20, 65), (50, 35), (30, 25), (15, 10), (30, 5), (10, 20), (5, 30), (20, 40), (15, 60), (45, 65), (45, 20), (45, 10), (55, 5), (65, 35), (65, 20), (45, 30), (35, 40), (41, 37), (64, 42), (40, 60), (31, 52), (35, 69), (53, 52), (65, 55), (63, 65), (2, 60), (20, 20), (5, 5), (60, 12), (40, 25), (42, 7), (24, 12), (23, 3), (11, 14), (6, 38), (2, 48), (8, 56), (13, 52), (6, 68), (47, 47), (49, 58), (27, 43), (37, 31), (57, 29), (63, 23), (53, 12), (32, 12), (36, 26), (21, 24), (17, 34), (12, 24), (24, 58), (27, 69), (15, 77), (62, 77), (49, 73), (67, 5), (56, 39), (37, 47), (37, 56), (57, 68), (47, 16), (44, 17), (46, 13), (49, 11), (49, 42), (53, 43), (61, 52), (57, 48), (56, 37), (55, 54), (15, 47), (14, 37), (11, 31), (16, 22), (4, 18), (28, 18), (26, 52), (26, 35), (31, 67), (15, 19), (22, 22), (18, 24), (26, 27), (25, 24), (22, 27), (25, 21), (19, 21), (20, 26), (18, 18)]

# 寻找最近的三个点
nearest_neighbors = find_k_nearest_neighbors(points, (0,0), 3)
print('最近的三个点及距离：', nearest_neighbors)




import numpy as np
#from vrp import *
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from typing import List, Tuple
import math
def find_k_nearest_neighbors(points: List[Tuple[float, float]], target_point: Tuple[float, float], k: int) -> List[
    Tuple[Tuple[float, float], float]]:
    # 构建kdtree
    #tree = KDTree(points)
    tree = KDTree(np.asarray(points, dtype=object))
    # 找k个最近邻
    dist, ind = tree.query([target_point], k=k)
    #print(ind)
    # print("nearest_k_points:", nearest_k_points)
    # 返回k个最近邻的坐标和距离
    return [(tuple(tree.data[i]), d) for i, d in zip(ind[0], dist[0])],ind


list=[[((35.0, 35.0), 0.0), ((41.0, 49.0), 10.0), ((20.0, 50.0), 5.0), ((55.0, 20.0), 19.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((30.0, 60.0), 16.0), ((20.0, 65.0), 12.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((55.0, 60.0), 16.0), ((50.0, 35.0), 19.0), ((35.0, 35.0), 0.0)], [((35.0, 35.0), 0.0), ((25.0, 30.0), 3.0), ((30.0, 25.0), 23.0), ((35.0, 17.0), 7.0), ((35.0, 35.0), 0.0)]]
lst=[]
for p in list:
    for p1 in p:
        lst.append(p1[0])
print(lst)
nearest_neighbors, ind1 = find_k_nearest_neighbors(lst[1:], lst[0], len(lst)-1)
for i in range(len(lst)-1):
     if(nearest_neighbors[i]!=((35.0, 35.0),0)):
         k=i
         break
print(k)
print(nearest_neighbors)
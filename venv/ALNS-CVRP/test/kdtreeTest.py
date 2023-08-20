import random
from kdtree import KDTree

# 生成随机数据集
data = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)]
print(data)
# 构建KDTree
tree = KDTree(data)
# 搜索最近的3个邻居点
target = (50, 50)
nearest_neighbors = tree.search_knn(target, 3)
print("The nearest neighbors of target point {} are: {}".format(target, nearest_neighbors))
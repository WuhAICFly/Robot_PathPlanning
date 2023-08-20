import numpy as np
from scipy.spatial import KDTree

# 构造一个 2D 数据集
data = np.random.rand(100, 2)

# 构建 KDTree
tree = KDTree(data)

# 定义查询点和参数
tx, ty = 0.5, 0.5
n = 7
distance_upper_bound = 0.03

# 查询距离该点最近的 k=3 个点，距离不超过 distance_upper_bound=3
dists, indexes = tree.query([tx, ty], k=n, distance_upper_bound=distance_upper_bound)

# 打印搜索结果
print("最近的 %d 个点的索引是：%s" % (n, indexes))
print("最近的 %d 个点距离查询点的距离是：%s" % (n, dists))

import matplotlib.pyplot as plt


plt.rc('font',family='Times New Roman')

#plt.rcParams['font.family'] = ['SimSun', 'Arial']  # 指定字体，支持中文和英文

class Node:
    def __init__(self, point, split, left, right):
        self.point = point
        self.split = split
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points:
        return None

    k = len(points[0])
    axis = depth % k

    sorted_points = sorted(points, key=lambda x: x[axis])
    mid = len(sorted_points) // 2

    return Node(
        sorted_points[mid],
        axis,
        build_kdtree(sorted_points[:mid], depth + 1),
        build_kdtree(sorted_points[mid + 1:], depth + 1)
    )

def plot_kdtree(node, min_x, max_x, min_y, max_y, ax, depth=0):
    if node is None:
        return

    k = len(node.point)
    axis = depth % k

    if axis == 0:
        ax.plot([node.point[0], node.point[0]], [min_y, max_y], '-k')
        plot_kdtree(node.left, min_x, node.point[0], min_y, max_y, ax, depth + 1)
        plot_kdtree(node.right, node.point[0], max_x, min_y, max_y, ax, depth + 1)
    else:
        ax.plot([min_x, max_x], [node.point[1], node.point[1]], '-k')
        plot_kdtree(node.left, min_x, max_x, min_y, node.point[1], ax, depth + 1)
        plot_kdtree(node.right, min_x, max_x, node.point[1], max_y, ax, depth + 1)

# Example usage
import matplotlib.pyplot as plt

points = [
    (round(4.108466572649949, 1), round(-3.791764796615399, 1)),
    (round(29.25760081238594, 1), round(0.7027335609583796, 1)),
    (round(19.770346896001584, 1), round(8.551106172543419, 1)),
    (round(18.636325505250085, 1), round(0.09343588875089537, 1)),
    (round(7.4293767904578365, 1), round(-8.010038804104383, 1)),
    (round(9.08965306122449, 1), round(20.74965306122449, 1)),
    (round(12.071118118779987, 1), round(17.413177864808503, 1)),
    (round(9.491149709046024, 1), round(14.48019749603245, 1)),
    (round(4.7653519355169935, 1), round(12.017968990656128, 1)),
    (round(8.030402630519394, 1), round(7.773812239967791, 1)),
    (round(11.376272813643613, 1), round(3.796528455475275, 1)),
    (round(7.245, 1), round(1.6983333333333337, 1)),
    (round(8.641694352159467, 1), round(11.143554817275746, 1)),
    (round(8.666037522404723, 1), round(17.5145701231158, 1)),
    (round(43.79294827358115, 1), round(10.288628257706044, 1))
]
# 设置全局字体大小
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots()
ax.scatter([x[0] for x in points], [x[1] for x in points])
for i, point in enumerate(points):
    ax.annotate(f' {point}', xy=point, xytext=(-15, 10), textcoords='offset points', fontsize=10, color='b')

tree = build_kdtree(points)
plot_kdtree(tree, min_x=0, max_x=45, min_y=-10, max_y=25, ax=ax)
plt.xlabel("x/m",fontdict={'size': 20})
plt.ylabel("y/m",fontdict={'size': 20})
# 显示图像
plt.legend(loc='lower right',fontsize=12)
plt.show()

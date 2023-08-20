import matplotlib.pyplot as plt

# 第一条线路的点坐标和容量
p1 = [((35.0, 35.0), 0.0), ((45.0, 30.0), 17.0), ((57.0, 29.0), 18.0), ((63.0, 23.0), 2.0), ((65.0, 20.0), 6.0),
      ((65.0, 35.0), 3.0), ((64.0, 42.0), 9.0), ((56.0, 39.0), 36.0), ((56.0, 37.0), 6.0), ((35.0, 35.0), 0.0)]

# 提取点坐标和容量
points = [p[0] for p in p1]
capacities = [p[1] for p in p1]
print(points)
# 绘制点和容量
fig, ax = plt.subplots()
ax.scatter(*zip(*points), s=50, color='blue')  # 绘制点
for i, point in enumerate(points):
    ax.annotate(capacities[i], point, textcoords="offset points", xytext=(0, 10), ha='center')  # 标注容量
plt.plot(*zip(*points), color='black')  # 绘制线路
plt.show()

#
# #线路的点坐标和容量
# # pos = [((35.0, 35.0), 0.0), ((45.0, 30.0), 17.0), ((57.0, 29.0), 18.0), ((63.0, 23.0), 2.0), ((65.0, 20.0), 6.0),
# #       ((65.0, 35.0), 3.0), ((64.0, 42.0), 9.0), ((56.0, 39.0), 36.0), ((56.0, 37.0), 6.0), ((35.0, 35.0), 0.0)]
# points=[]
# capacities=[]
# for p1 in pos:
#   #print(p1)
#
#   # 提取点坐标和容量
#   point = [p[0] for p in p1]
#   capacitie = [p[1] for p in p1]
#   points.append(point)
#   capacities.append(capacitie)
#   print(points[0])
#   points = points[0]
# # 绘制点和容量
#   fig, ax = plt.subplots()
#   ax.scatter(*zip(*points), s=50, color='blue')  # 绘制点
#   # for i, point in enumerate(points):
#   #   ax.annotate(capacities[i], point, textcoords="offset points", xytext=(0, 10), ha='center')  # 标注容量
#   plt.plot(*zip(*points), color='black')  # 绘制线路
#   plt.show()



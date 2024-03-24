import math
import matplotlib.pyplot as plt

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def interpolate_points(points):
    result = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        # 计算线段的斜率和截距
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - k * p1[0]
        # 沿着x轴方向取点
        for x in range(p1[0], p2[0] + 1):
            y = k * x + b
            if y == int(y):
                result.append((int(x), y))
            else:
                result.append((int(x), y))
        # 将线段的终点作为插值点
        # if i == len(points) - 2:
        #     result.append(p2)
    return result

points = [(126,178),(238, 106)]
interpolated_points = interpolate_points(points)

# for point in interpolated_points:
#     print(point)
print(interpolated_points)


with open("y_values25", "w") as f:
    for point in interpolated_points:
        f.write(f"{point[0]} {point[1]}\n")
# 绘制结果
plt.plot([p[0] for p in points], [p[1] for p in points], c='b', label='Original points')
plt.scatter([p[0] for p in interpolated_points], [p[1] for p in interpolated_points], c='r', label='Interpolated points')
plt.legend()
plt.show()



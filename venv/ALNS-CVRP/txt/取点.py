import math
import matplotlib.pyplot as plt

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def interpolate_points(points, step=1):
    result = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        dist = distance(p1, p2)
        steps = int(dist // step)
        for j in range(steps + 1):
            x = p1[0] + (p2[0] - p1[0]) * j / steps
            y = p1[1] + (p2[1] - p1[1]) * j / steps
            result.append((int(x), y))
    return result

points = [(126,178),(184, 291),(304, 329)]
interpolated_points = interpolate_points(points)

#for point in interpolated_points:
print(interpolated_points)


with open("y_values25", "w") as f:
    for point in interpolated_points:
        f.write(f"{point[0]} {point[1]}\n")

# 绘制结果
plt.plot([p[0] for p in points], [p[1] for p in points], c='b', label='Original points')
plt.scatter([p[0] for p in interpolated_points], [p[1] for p in interpolated_points], c='r', label='Interpolated points')
plt.legend()
plt.show()

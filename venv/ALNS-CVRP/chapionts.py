import math
import matplotlib.pyplot as plt
def insert_points(points):
    new_points = [points[0]]
    for i in range(1, len(points)-1):
        p1 = points[i-1]
        p2 = points[i]
        p3 = points[i+1]
        angle = math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        angle = angle * 180 / math.pi
        if angle > 90:
            print(i,angle)
            x1, y1 = p2
            x2, y2 = p3
            x, y = (x1+x2)/2, (y1+y2)/2
            new_points.append((x1, y1))
            new_points.append((x, y))
            new_points.append((x2, y2))
        else:
            new_points.append(p2)
    new_points.append(points[-1])
    return new_points


# Example usage
points = [(0, 0), (1, 0), (2, 1), (3, 3), (4, 4), (5, 5), (6, 6), (7, 6), (8, 6)]
new_points = insert_points(points)
rx=[]
ry=[]
for points in new_points:
    rx.append(points[0])  # 取出x坐标并添加到x_coords列表中
    ry.append(points[1])
plt.plot(rx,ry)
plt.show()
print(rx)
print(ry)
print(new_points)





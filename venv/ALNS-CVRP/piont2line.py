import math

point1 = (-5, 10)
point2 = (55, 10)

A = point2[1] - point1[1]
B = point1[0] - point2[0]
C = point2[0] * point1[1] - point1[0] * point2[1]

x0, y0 = 1, 7
d = abs(A*x0 + B*y0 + C) / math.sqrt(A**2 + B**2)
print("点（0,0）到直线的距离为：", d)

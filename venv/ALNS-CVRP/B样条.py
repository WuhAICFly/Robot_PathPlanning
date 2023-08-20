import numpy as np
import matplotlib.pyplot as plt

def b_spline(t, i, k, t_list):
    """
    计算B样条基函数值
    """
    if k == 0:
        if t_list[i] <= t < t_list[i+1]:
            return 1
        else:
            return 0
    else:
        w1 = 0
        if t_list[i+k] != t_list[i]:
            w1 = (t - t_list[i]) / (t_list[i+k] - t_list[i]) * b_spline(t, i, k-1, t_list)
        w2 = 0
        if t_list[i+k+1] != t_list[i+1]:
            w2 = (t_list[i+k+1] - t) / (t_list[i+k+1] - t_list[i+1]) * b_spline(t, i+1, k-1, t_list)
        return w1 + w2

def b_spline_curve(control_points, num_points):
    """
    计算B样条曲线上的点
    """
    n = len(control_points)
    k = 3
    t_list = np.linspace(0, 1, n+k+1)
    result_points = []
    # 计算每个参数对应的点坐标
    for t in np.linspace(0, 1, num_points):
        x, y = 0, 0
        for i in range(n):
            b = b_spline(t, i, k, t_list)
            x += control_points[i][0] * b
            y += control_points[i][1] * b
        result_points.append((x, y))
    return result_points

# 测试代码
control_points = [(0, 0), (1, 3), (2, -1), (3, 4)]
num_points = 100
result_points = b_spline_curve(control_points, num_points)

# 绘制B样条曲线
x = [p[0] for p in result_points]
y = [p[1] for p in result_points]
plt.plot(x, y, 'b-', label="B-spline curve")

# 绘制控制点
x = [p[0] for p in control_points]
y = [p[1] for p in control_points]
plt.plot(x, y, 'ro-', label="Control points")

plt.legend()
plt.show()

import cv2
import numpy as np

# 读取图片
img = cv2.imread('image.png')

# 定义坐标轴的起始和结束点
origin = (0, img.shape[0])
x_axis_end = (img.shape[1], img.shape[0])
y_axis_end = (0, 0)

# 绘制坐标轴
cv2.line(img, origin, x_axis_end, (0, 0, 0), 2)
cv2.line(img, origin, y_axis_end, (0, 0, 0), 2)

# 计算并绘制坐标值
for x in range(0, img.shape[1], 50):
    cv2.putText(img, str(x), (x, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
for y in range(0, img.shape[0], 50):
    cv2.putText(img, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# 显示图片
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

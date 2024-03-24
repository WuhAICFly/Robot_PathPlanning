import cv2
import numpy as np

# 读取图片
img = cv2.imread('image.png')

# 创建一个回调函数
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"坐标：({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

# 显示图片
cv2.imshow('Image', img)

# 设置鼠标事件回调
cv2.setMouseCallback('Image', draw_circle)

cv2.waitKey(0)
cv2.destroyAllWindows()

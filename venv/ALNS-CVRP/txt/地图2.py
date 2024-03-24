import cv2
import numpy as np

# 读取图片
img = cv2.imread('image.png')

# 检查图片是否正确加载
if img is None:
    print("图片没有正确加载，请检查图片文件路径和格式。")
else:
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Thresholded Image', gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

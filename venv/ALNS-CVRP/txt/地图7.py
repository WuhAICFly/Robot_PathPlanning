import matplotlib.pyplot as plt
import matplotlib.image as imread
import numpy as np

# 读取图片
image_path = 'image1.png'
img = imread.imread(image_path)

# 将图片水平翻转
img = np.fliplr(img)
# 将图片旋转180度
img = np.rot90(img, 2)
# 显示图片
plt.imshow(img, cmap='gray')

# 设置标题
plt.title('Image')

# 显示坐标轴
plt.axis('on')

# 绘制两点之间的线
#plt.plot([63, 221], [281, 171], color='red', linewidth=2)

# 反转y轴，使0刻度位于左下方
plt.gca().invert_yaxis()

# 设置y轴范围，使0刻度位于左下方
#plt.xlim([387000, 388000])
#plt.ylim([3110460, 3111520])
# 显示图片
plt.show()

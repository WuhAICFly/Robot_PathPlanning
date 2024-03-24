import matplotlib.pyplot as plt
import matplotlib.image as imread

# 读取图片
image_path = 'image1.png'
img = imread.imread(image_path)

# 显示图片
plt.imshow(img)

# 添加标题
plt.title('Image')

# 显示坐标轴
plt.axis('on')
# 绘制两点之间的线
plt.plot([63, 220], [302, 415], color='red', linewidth=2)
# 显示图片
plt.show()

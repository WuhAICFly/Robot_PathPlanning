import matplotlib.pyplot as plt

def cut(s):
    w1_str = s.split('w')[1]
    w2_str = s.split('w')[2]
    if w1_str[0] == '0':
        w1 = '0.' + w1_str[1:]
    else:
        w1 = w1_str

    if w2_str[0] == '0':
        w2 = '0.' + w2_str[1:]
    else:
        w2 = w2_str

    return w1, w2


list=['w100w0_01','w100w0_1','w100w1','w100w10','w100w100','w100w1000']
yy=[]
ww=[]
for i in range(6):
  k_QP = 'k_QP'
  w = list[i]
  file_name = k_QP + w + '.txt'
  s = w.replace('_', '')
  print(s)
  w1,w2=cut(s)
  with open(file_name, 'r') as file:
    points = []
    for line in file:
        # 将每行数据按空格分隔，转为浮点数后添加到points列表中
        points.append([float(coord) for coord in line.split()])


  x  = [item[0] for item in points]
  y = [item[1] for item in points]
  print(y)
  yy.append(y)
  ww.append(w2)
print(yy)


# 绘制线图并添加标注
plt.plot(x, yy[0],label='w2='+str(ww[0]))
plt.plot(x, yy[1],label='w2='+str(ww[1]))
plt.plot(x, yy[2],label='w2='+str(ww[2]))
plt.plot(x, yy[3],label='w2='+str(ww[3]))
plt.plot(x, yy[4],label='w2='+str(ww[4]))
plt.plot(x, yy[5],label='w2='+str(ww[5]))
# 添加标题和坐标轴标签以及图例
plt.title('w1='+str(w1))
plt.xlabel("s")
plt.ylabel("m/s")
plt.legend()
plt.show()

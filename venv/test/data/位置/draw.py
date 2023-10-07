import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
rx=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
ry=[10,11.2394606,12.42899247,13.40048004,14.37196761,14.99377239,15.56817888,16.14258538,
16.71699188,17.29139837,20.39822242,23.80519111,24.16959558,24.3110094,24.45242322,24.59383704,24.73525086,
24.87666468,25.0180785,25.15949232,21.23042532,20.58113894,19.93185255,18.29863972,17.06376829,16.2021173,15.3404663,
14.00768861,12.4978392,9.556195843,5
]
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
  S_QP = 'S_QP'
  w = list[i]
  file_name = S_QP + w + '.txt'
  s = w.replace('_', '')
  #print(s)
  w1,w2=cut(s)
  with open(file_name, 'r') as file:
    points = []
    for line in file:
        # 将每行数据按空格分隔，转为浮点数后添加到points列表中
        points.append([float(coord) for coord in line.split()])


  x  = [item[0] for item in points]
  xx=  [item[0]-5 for item in points]
  y = [item[1] for item in points]
  #print(y)
  yy.append(y)
  ww.append(w2)
#print(yy)

print(rx)
print(ry)
print(x)
print(xx)
print(yy[5])
# 绘制线图并添加标注
plt.plot(rx, ry,  label='Original Trajectory')
#plt.plot(x, yy[0],label='Smooth trajectory')
#plt.plot(x, yy[1],label='Smooth trajectory')
#plt.plot(x, yy[2],label='Smooth trajectory')
#plt.plot(x, yy[3],label='Smooth trajectory')
#plt.plot(x, yy[4],label='Smooth trajectory')
plt.plot(x, yy[5],label='Smooth trajectory')
# 添加标题和坐标轴标签以及图例
plt.title('w1='+str(w1))
#plt.xlabel("time(s)")
#plt.ylabel(" Smooth trajectory(m)")
plt.legend()
plt.show()

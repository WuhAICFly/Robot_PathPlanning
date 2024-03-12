import numpy as np
import copy
import random
import math
import time
import matplotlib.pyplot as plt



###step 1 数据处理
##数据读入并转换为np.array
cor_XY = np.genfromtxt('data.csv', delimiter=',')
##距离矩阵,矩阵计算
num_data = len(cor_XY)
cor_XY_2 = cor_XY * cor_XY
XY_sqrtsum = [cor_XY_2[i,0] + cor_XY_2[i,1] for i in range(num_data)]
XX = np.array([XY_sqrtsum for i in range(num_data)])
YY = XX.T
XY = np.dot(cor_XY,cor_XY.T)
D_XY = np.sqrt(XX + YY - 2 * XY)

###step 2 初始化
#指标计算函数
def L_route(route):
    L_route_now = 0
    for i in range(num_data-1):
        L_route_now += D_XY[route[i],route[i+1]]
    L_route_now += D_XY[route[-1],route[0]]
    return L_route_now
def destroy(route):
    route_f = copy.deepcopy(route)
    delete = random.choice(route_f)
    route_f.remove(delete)
    return route_f,delete
def repair(route_d,insert,index):
    route_f = copy.deepcopy(route_d)
    route_f.insert(index,insert)
    return route_f
###Step 3 LNS算法
#生成初始可行解
route_o = [i for i in range(num_data)]
random.shuffle(route_o)
#初始化
route_best = route_o
route = route_o
L_route_best = L_route(route_o)
L_min_list = []
L_min_list.append(L_route_best)
N = 20000#设置迭代次数
#迭代
t_s = time.time()
print('进度：0  ',end = '')
for i in range(N):
    route_d,delete = destroy(route)
    N_repairs = []
    L_repairs = []
    for index in range(len(route_best) - 1):
        route_f = repair(route_d,delete,index)
        N_repairs.append(route_f)
        L_repairs.append(L_route(route_f))
    L_repairs_min = min(L_repairs)
    index_min = L_repairs.index(L_repairs_min)
    route_repairs_min = N_repairs[index_min]
    if L_repairs_min < L_route(route):
        route = route_repairs_min
    if L_repairs_min < L_route_best:
        route_best = route
        L_route_best = L_repairs_min
        L_min_list.append(L_route_best)
    #进度指示
    if (i + 1) % 2000 == 0:
        print((i + 1) / 200,'%',end = '  ')
print('迭代结束')
t_e = time.time()
t_p = t_e - t_s

###step 4 输出
##结果日志
txt_name = '结果日志.txt'
with open(txt_name,'w') as f:
    f.write('最短距离为：%s\n'%L_route_best)
    f.write('求解时间为：%s S\n'%t_p)
    f.write('路线为：%s \n'%route_best)
##绘图
#绘图1：结果路线图
plt.figure()
plt.scatter(cor_XY[:,0],cor_XY[:,1])
for i in range(num_data - 1):
    x_list = [cor_XY[route_best[i]][0],cor_XY[route_best[i + 1]][0]]
    y_list = [cor_XY[route_best[i]][1],cor_XY[route_best[i + 1]][1]]
    plt.plot(x_list,y_list)
x_list = [cor_XY[route_best[num_data - 1]][0],cor_XY[route_best[0]][0]]
y_list = [cor_XY[route_best[num_data - 1]][1],cor_XY[route_best[0]][1]]
plt.plot(x_list,y_list)
plt.savefig('路线图.png',dpi = 600,format='png')#矢量图
plt.show()
#绘图2：结果趋势图
plt.figure()
x_lists = list(range(len(L_min_list)))
y_lists = L_min_list
plt.plot(x_lists,y_lists)
plt.savefig('结果趋势图.png',dpi = 600,format='png')#矢量图
plt.show()
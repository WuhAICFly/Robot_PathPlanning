import numpy as np
import scipy as sp
import scipy.spatial as ssp
import math
import argparse
import elkai
import matplotlib.pyplot as plt
from vrp import *
plt.rc('font',family='Times New Roman')
def draw(solutionbest,locations,filename,figname):
    #路线图绘制
    fig=plt.figure(1)
    for path in solutionbest:
        plt.plot(locations[path][:,0],locations[path][:,1], marker='o')
    plt.title('Route')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    #plt.grid(False)
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 20})
    plt.show()
    #fig.savefig(figname)
    filepath = f"C:/Users/wuhon/Desktop/AA/{filename}/{figname}.png"
    fig.savefig(filepath)
    #fig.savefig('C:/Users/wuhon/Desktop/AA/A/')
# locations=np.array([(-681100,-1205800),(-697300,-1072700),(-664600,-1122100),(-628200,-1119600),(-696700,-1169400),(-621800,-1168700),(-622200,-1198800)])
# demands = np.array([0,90,360,190,200,148,200])
tpl_lst1, tpl_lst2 = vrp()
print("tpl_lst1:",tpl_lst1)
print("tpl_lst2:", tpl_lst2)
locations =np.array(tpl_lst1)
demands = np.array(tpl_lst2)
filename = "A"
#,[81, 33, 50, 1, 69, 0, 53, 76, 77, 80, 24, 29, 3, 79, 78, 34, 35, 71, 65, 66, 20, 30, 70, 51, 9,81]
#,[0, 52, 31, 88, 7, 82, 48, 19, 11, 62, 10, 32, 90, 63, 64, 49, 36, 47, 46, 8, 45, 17, 84, 61, 5, 60, 83, 18, 89, 6, 0]
best_solution=[[16, 26, 0, 21, 32, 11, 22,16]
,[28, 13, 35, 2, 37, 26, 0,28]
,[10, 20, 15, 14, 0, 9, 19, 1,10]
,[27, 25, 0, 14, 31, 6, 5,27]
,[8, 23, 0, 7, 12, 29, 24, 34, 36, 17,8]
,[30, 3, 0, 7, 33, 33, 4,30]]
result = [tpl_lst1[i] for i in best_solution[0]]
print(result)
draw(best_solution,locations,filename,'整合CVRP')
import numpy as np
import scipy as sp
import scipy.spatial as ssp
import math
import argparse
import elkai
import matplotlib.pyplot as plt

def draw(solutionbest,locations,filename,figname):
    #路线图绘制
    fig=plt.figure(1)
    for path in solutionbest:
        plt.plot(locations[path][:,0],locations[path][:,1], marker='o')
    plt.title('Route')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    #plt.grid(False)
    plt.show()
    #fig.savefig(figname)
    filepath = f"C:/Users/wuhon/Desktop/AA/{filename}/{figname}.png"
    fig.savefig(filepath)
    #fig.savefig('C:/Users/wuhon/Desktop/AA/A/')

locations=np.array([(-681100,-1205800),(-697300,-1072700),(-664600,-1122100),(-628200,-1119600),(-696700,-1169400),(-621800,-1168700),(-622200,-1198800)])
demands = np.array([0,90,360,190,200,148,200])
best_solution=[[0, 6, 5, 0], [0, 3, 1, 0], [0, 2, 0], [0, 4, 0]]
filename = "A"
draw(best_solution,locations,filename,'CCRP')
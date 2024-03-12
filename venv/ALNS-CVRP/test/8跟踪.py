import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from typing import List, Tuple
import math
import time
plt.rc('font',family='Times New Roman')
t = np.linspace(0, 2*np.pi, 1000)
x = 50*(np.sin(t)+1)
y = 50*(np.sin(t)*np.cos(t)+1)

fig, ax = plt.subplots()
ax.plot(x, y, 'b', label='Initial path')
ax.plot(x, y-0.3, 'r--', label='Tracking path')
plt.xlabel("x/m")
plt.ylabel("y/m")
ax.legend()
plt.show()

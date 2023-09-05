import numpy as np

points = np.array([[35.0, 35.0], [21.0, 24.0], [25.0, 21.0], [28.0, 18.0], [30.0, 25.0]])

distmat = np.zeros((len(points), len(points)))

for i in range(len(points)):
    for j in range(len(points)):
        distmat[i][j] = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)

print(distmat)

import math
def cvrp( ):
    points = [(-5,10), (33.2,9.9), (35.2,20.5), (39.9,34.0), (51.4,-1.1), (-9.3,8.9), (30,10), (33,13)]
    indices1 = [[0, 5, 4, 1, 0], [0, 6, 0], [0, 7, 0], [0, 2, 3, 0]]
    indices2 = [[0, 6, 2, 3, 0], [0, 6, 1, 4, 5,0],[0, 7, 0]]
    indices=[indices1,indices2]
    pos=[]
    for i in range(2):
        new_points = [[points[i] for i in index] for index in indices[i]]

        pos.append(new_points)
    print(pos)
    return  pos
def quchu(k,points):
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        x1,y1=p1
        x2,y2=p2
        p=(x1,y1,x2,y2)
        p = tuple(round(num) for num in p)
        print(k,p)
        p=[k,p]
        with open('pos.txt', 'a') as f:
            data = str(p)
            f.write(data)
            f.write('\n')
def qupos(pos):
    k=-1
    for p in pos:
        k=k+1
        for p in p:
            quchu(k,p)

pos=cvrp()
qupos(pos)




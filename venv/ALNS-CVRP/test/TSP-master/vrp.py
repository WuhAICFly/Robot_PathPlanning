coords = []

def vrp():
 with open('Golden_2.txt') as f:
    lines = f.readlines()
    for line in lines:
        # 按空格分割字符串，获取坐标
        coord_str = line.strip().split(' ')[1:]
        # 将字符串坐标转换为浮点数
        coord = [float(c) for c in coord_str]
        coords.append(coord)
 lst1 = [i for i in coords[:coords.index([])] if i != []]
 lst2 = [i for i in coords[coords.index([])+1:] if i != []]

 tpl_lst1 = [(x[0], x[1]) for x in lst1]
 tpl_lst2 = [(x[0]) for x in lst2]
 return tpl_lst1 ,tpl_lst2

tpl_lst1,tpl_lst2=vrp()

# print(tpl_lst1)
# print(tpl_lst2)





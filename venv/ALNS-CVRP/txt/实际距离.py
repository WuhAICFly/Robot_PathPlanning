def find_point_index(coords, point):
    for i in range(len(coords)):
        if coords[i] == point:
            return i
    return None

def count_points_between_points(coords, p1, p2):
    count = 0
    for i in range(len(coords)):
        if i != p1 and i != p2:
            x, y = coords[i]
            x1, y1 = coords[p1]
            x2, y2 = coords[p2]
            if (x2 - x1) != 0 and (y2 - y1) != 0:
                if (x - x1) / (x2 - x1) == (y - y1) / (y2 - y1):
                    count += 1
    return count

coords = [(-681100,-1205800),(-664600,-1122100),(-628200,-1119600)]
p1 = (-681100,-1205800)
p2 = (-628200,-1119600)

p1_index = find_point_index(coords, p1)
p2_index = find_point_index(coords, p2)

if p1_index is not None and p2_index is not None:
    count = count_points_between_points(coords, p1_index, p2_index)
    print(f"There are {count} points between points {p1} and {p2}.")
else:
    print("One or both points not found in coords.")

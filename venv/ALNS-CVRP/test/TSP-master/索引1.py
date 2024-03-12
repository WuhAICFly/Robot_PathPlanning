list1 = [(387405,3110653),(387541,3111438),(387463,3111265),(387638,3111168),(387573,3111029),(387135,3111127),(387325,3111025),(388035,3110696)]

list3=[((387405, 3110653), 0), ((388035, 3110696), 305)]
lst=[p[0] for p in list3]
print(lst)
with open('output1.txt', 'w') as f:
    for i, item in enumerate(lst):
        f.write(f'{i+1}\t{item}\n')

# 定义原始列表
#list = [(37.0, 47.0), (41.0, 49.0), (47.0, 47.0), (49.0, 42.0), (53.0, 43.0), (55.0, 45.0), (57.0, 48.0), (53.0, 52.0), (55.0, 54.0), (55.0, 60.0), (49.0, 58.0), (37.0, 56.0), (40.0, 60.0), (45.0, 65.0), (49.0, 73.0), (62.0, 77.0), (57.0, 68.0), (63.0, 65.0), (65.0, 55.0), (61.0, 52.0), (64.0, 42.0), (65.0, 35.0), (56.0, 37.0), (35.0, 35.0), (37.0, 31.0)]

# 定义要排序的索引列表
indices = [1, 0]
#indices = [7, 1, 0, 23, 24, 22, 21, 20, 5, 4, 3, 2, 13, 14, 15, 16, 17, 18, 19, 6, 11, 12, 10, 9, 8]
# 使用sorted()函数和lambda表达式按照索引排序
result = [lst[i] for i in indices]
print(result)



list2=result
# Store the indices of the matching elements in list1
indices = {list1[i]: i for i in range(len(list1)) if list1[i] in list2}

# Find the indices of the matching elements in list2
result = [indices[elem] for elem in list2 if elem in indices]

print(result)  # Output: [0, 7, 8]
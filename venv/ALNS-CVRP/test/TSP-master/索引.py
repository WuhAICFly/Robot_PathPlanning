lst1 = [(387405,3110653),(387541,3111438),(387463,3111265),(387638,3111168),(387573,3111029),(387135,3111127),(387325,3111025),(388035,3110696)]
# list2=[(45.0, 30.0), (57.0, 29.0), (55.0, 20.0)]
list3=[((387405, 3110653), 0), ((387638, 3111168), 190), ((387541, 3111438), 220)]
list=[p[0] for p in list3]
print(list)
# list2=lst
# # Store the indices of the matching elements in list1
# indices = {list1[i]: i for i in range(len(list1)) if list1[i] in list2}
#
# # Find the indices of the matching elements in list2
# result = [indices[elem] for elem in list2 if elem in indices]
#
# print(result)  # Output: [0, 7, 8]

with open('output3.txt', 'w') as f:
    for item in list:
        f.write(f'{item[0]}, {item[1]}\n')

# with open('output.txt', 'w') as f:
#     for i, item in enumerate(list, 1):
#         f.write(f'{i} {item[0]},{item[1]}\n')

with open('output2.txt', 'w') as f:
    for i, item in enumerate(list):
        f.write(f'{i+1}\t{item}\n')





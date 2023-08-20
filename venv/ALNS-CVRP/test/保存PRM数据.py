filename = 'output.txt' # 可以替换为您要写入的文件名
data1=1
data2=2
data3=3
with open(filename, 'a') as f:
    f.write(str(data1))
    f.write(' ')

    f.write(str(data2))
    f.write(' ')

    f.write(str(data3))
    f.write('\n')

# # 导入chardet库
# import chardet
#
# # 检测文件编码
# with open("a.txt", "rb") as file:
#    result = chardet.detect(file.read())
#
# # 打开文件并读取所有行
# with open("a.txt", "r", encoding=result["encoding"]) as file:
#    lines = file.readlines()
# print(lines)
sspos=[]
ffpos=[]
#打开文件并读取所有行
with open("a.txt", "r") as file:
  lines = file.readlines()
  print(lines)
  print(len(lines))
# 遍历每一行并处理
for line in lines:
  # 将字符串转换为列表
  list = eval(line.strip())
  # 遍历列表中的点并赋值给spos和fpos
  spos = None
  fpos = list[0]

  for i in range(1, len(list)):
      spos = fpos
      fpos = list[i]

  # 打印结果
      print("i=:",i)
      print("spos:", spos)
      sspos.append(spos)
      print("fpos:", fpos)
      ffpos.append(fpos)
print(len(sspos))
print("list1=",sspos)
print("list2=",ffpos)
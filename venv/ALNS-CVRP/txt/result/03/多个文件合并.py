import os

# 获取当前目录下的所有txt文件
#txt_files = [f for f in os.listdir() if f.endswith('.txt')]
# 指定要合并的txt文件列表
txt_files = ['v_QP1.txt', 'v_QP2.txt']
# 创建一个新的txt文件，用于存储合并后的内容
with open('v_QP.txt', 'w', encoding='utf-8') as merged_file:
    # 遍历所有txt文件
    for txt_file in txt_files:
        # 读取txt文件内容
        with open(txt_file, 'r', encoding='utf-8') as file:
            content = file.read()
            # 将内容写入新的txt文件
            merged_file.write(content)
            # 添加换行符，以便在合并后的文件中分隔每个文件的内容
            #merged_file.write('\n')

print('合并完成！')

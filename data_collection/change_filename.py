import os

# path = input('请输入文件路径(结尾加上/)：')
path ='dataset_val'
# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)

print("fileList:",fileList)
# n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + i  # os.sep添加系统分隔符

    # 设置新文件名
    newname = path + os.sep  +str(int(i[:-4]) + 2000) + '.npz'

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    # n += 1
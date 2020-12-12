import random
set_val=set()

for i in range(2000,4000):
    set_val.add(i)

print(len(set_val))
print(set_val)

lst = [1,2,3]
if 3 in lst:
    print("yes")




#2、利用Python中的randomw.sample()函数实现
resultList=random.sample(range(2050,2499),50); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
print(len(resultList))# 打印结果
print(resultList)# 打印结果

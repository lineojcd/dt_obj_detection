import random
ranlst_1=[]
ranlst_2=[]
ranlst_3=[]
ranlst_4=[]
set_val=set()
set_whole=set()
set_train=set()

# for i in range(50):
#     if random.randint(2050,2499) not in ranlst_1:
#         ranlst_1.append(random.randint(2050,2499))
#     ranlst_2.append(random.randint(2550,2999))
#     ranlst_3.append(random.randint(3050,3499))
#     ranlst_4.append(random.randint(3550,3999))

#2、利用Python中的randomw.sample()函数实现
ranlst_1=random.sample(range(2050,2500),50); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_2=random.sample(range(2550,3000),50); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_3=random.sample(range(3050,3500),50); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_4=random.sample(range(3550,4000),50); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。

# for i in range(12):
#     print(random.randint(2050,2052))
print(len(ranlst_1))
print(ranlst_1)
print(len(ranlst_2))
print(ranlst_2)
print(len(ranlst_3))
print(ranlst_3)
print(len(ranlst_4))
print(ranlst_4)
for i in ranlst_1:
    set_val.add(i)
for i in ranlst_2:
    set_val.add(i)
for i in ranlst_3:
    set_val.add(i)
for i in ranlst_4:
    set_val.add(i)
for i in range(2000,4000):
    set_whole.add(i)
set_train = set_whole - set_val
print('len(set_val)',len(set_val))
print(set_val)
print('len(set_train)',len(set_train))
print(set_train)
print('len(set_whole)',len(set_whole))
print(set_whole)
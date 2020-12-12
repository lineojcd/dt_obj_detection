import json
import random
ranlst_1=[]
ranlst_2=[]
ranlst_3=[]
ranlst_4=[]
set_val=set()
set_whole=set()
set_train=set()

#2、利用Python中的randomw.sample()函数实现
ranlst_1=random.sample(range(2050,2500),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_2=random.sample(range(2550,3000),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_3=random.sample(range(3050,3500),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_4=random.sample(range(3550,4000),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
for i in ranlst_1:
    set_val.add(i)
for i in ranlst_2:
    set_val.add(i)
for i in ranlst_3:
    set_val.add(i)
for i in ranlst_4:
    set_val.add(i)
for i in range(2050,2500):
    set_whole.add(i)
for i in range(2550,3000):
    set_whole.add(i)
for i in range(3050,3500):
    set_whole.add(i)
for i in range(3550,4000):
    set_whole.add(i)

set_train = set_whole - set_val
print('len(set_val)',len(set_val))
print(set_val)
print('len(set_train)',len(set_train))
print(set_train)
print('len(set_whole)',len(set_whole))
print(set_whole)

ranlst_1a=[]
ranlst_2a=[]
ranlst_3a=[]
ranlst_4a=[]
set_val_a=set()
set_whole_a=set()
set_train_a=set()

ranlst_1a=random.sample(range(0,450),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_2a=random.sample(range(500,950),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_3a=random.sample(range(1000,1450),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
ranlst_4a=random.sample(range(1500,1950),75); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。

for i in ranlst_1a:
    set_val_a.add(i)
for i in ranlst_2a:
    set_val_a.add(i)
for i in ranlst_3a:
    set_val_a.add(i)
for i in ranlst_4a:
    set_val_a.add(i)
for i in range(0,450):
    set_whole_a.add(i)
for i in range(500,950):
    set_whole_a.add(i)
for i in range(1000,1450):
    set_whole_a.add(i)
for i in range(1500,1950):
    set_whole_a.add(i)
set_train_a = set_whole_a - set_val_a
print('len(set_val_a)',len(set_val_a))
print(set_val_a)
print('len(set_train_a)',len(set_train_a))
print(set_train_a)
print('len(set_whole_a)',len(set_whole_a))
print(set_whole_a)

set_train_combine = set_train_a | set_train
set_val_combine = set_val_a | set_val
set_whole_combine = set_whole_a | set_whole

print('len(set_train_combine)',len(set_train_combine))
print(set_train_combine)
print('len(set_val_combine)',len(set_val_combine))
print(set_val_combine)
print('len(set_whole_combine)',len(set_whole_combine))
print(set_whole_combine)

# Python 字典类型转换为 JSON 对象
data = {
    'train': list(set_train_combine),
    'val': list(set_val_combine)
}
json_str = json.dumps(data)
# print("Python 原始数据：", repr(data))
print("JSON 对象：", json_str)

print('intersection is',set_train_combine & set_val_combine)
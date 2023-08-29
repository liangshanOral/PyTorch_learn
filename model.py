#init(),用来构建子模块，forward，用来构建计算图返回计算结果

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
'''
class my_model(torch.nn.module):
    def __init__(self):
        super(my_model,self).__init__() #调用父类构造函数
        self.conv1=torch.nn.Conv2d(3,32,3,1,1)
        self.relu1=torch.nn.RELU() #这种不需要学习参数的也可以不在init中写出来
        self.max_pooling1=torch.nn.MaxPool2d(2,1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=F.relu(x)
        x=F.max_pool2d(x)
        x=self.max_pooling1(x)

        return x
    
model=my_model()
print(model)

#以上就是最简单的顺序连接，当然还可以通过构造模块等进行定义

class my_model2(torch.nn.Module):
    def __init__(self):
        super(my_model2,self).__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(), #记得打逗号啊
            nn.MaxPool2d(2)
        )
        self.dense_block=nn.Sequential(
            nn.Linear(32*3*3,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        conv_out=self.conv_block(x)
        res=conv_out.view(conv_out.size(0),-1) #result 所以一维排列把结果输出
        out=self.dense_block(res)

        return out

model2=my_model2()
print(model2)

#通过sequential来包装层

class my_model3(nn.Module):
    def __init__(self):
        super(my_model3,self).__init__()
        self.conv_block==nn.Sequential(
            OrderedDict(
            [
                ("conv1",nn.Conv2d(3,32,3,1,1))
            ]
            )
        )

        self.dense_block=torch.nn.Sequential()
        self.dense_block.add_module("dense1",torch.nn.Linear(32*3*3),128)

    def forward(self,x):
        #same
        return x
#this two way 每一层是有名字的

#sequential实现了整数索引，所以可以通过model[index]来获取层，但是module类不能这样，它通过返回iterator迭代器得到

for i in model2.children():
    print(i)
    print(type(i))

#children/named_children 返回的就是模型的子集，并不会层层深入
#modules/named_modules() 返回的是模型的所有构成，层层深入

#一些和module相关的功能


#1 除了子节点，其他中间变量的梯度都会被内存释放，为了获取他们，可以使用hook机制
a = torch.Tensor([1,2]).requires_grad_() 
b = torch.Tensor([3,4]).requires_grad_() 
e=a+b

def hook_fn(grad):
    print(grad)

e.register_hook(hook_fn)

# 用来获取模块的输出特征图和特征图梯度
register_forward_hook(module,input,output):
    options
register_backward_hook(module,grad_in,grad_out):
    options
# ref https://www.jianshu.com/p/69e57e3526b3
'''
'''
#create
empty_tensor=torch.empty(2,3)
data=[1,2,3]
data_tensor=torch.tensor(data)
#manipulate
matrix1=torch.mm(empty_tensor,data_tensor)
reshaped_tensor=matrix1.view(1,-1)

x=torch.tensor([2.0],requires_grad=True)
y=x ** 2
y.backward()
gradient=x.grad

#创建一个模型参数
param=nn.Parameter(torch.tensor([0.6],requires_grad=True))
#在module中定义参数
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel,self).__init__()
        self.weight=nn.Parameter(torch.zeros(10,5))
        #在model子类中使用parameter创建参数时，参数会自动注册成该model的一部分
        self.register_buffer("name",instance)

    def forward(self,x):
        output=torch.matmul(x,self.weight)
        return output
    
models=mymodel()
params=models.parameters()
'''
#linear
#参数在训练的时候自动学习，只需要定义输入输出维就好了
#input_features batch,dimension of one sample. out_features 你想输出的神经元个数
#相当于进行一个矩阵乘法

# 创建一个模型
model = nn.Linear(in_features=2, out_features=1)
input_data=torch.tensor([[0.1,0.2],[0.2,0.3]])
predictions=model(input_data)
predictions=F.relu(predictions) #sigmoid
print(predictions)

print(model.parameters)
print(model.state_dict()) #返回一个包含model参数的字典

#定义完了模型之后，就需要进行loss和update
Loss=nn.MSELoss() #还有很多种loss
'''size_average（默认为 True）：是否对损失进行平均。如果为 True，损失将被除以输入的数量，得到平均损失。如果为 False，损失将保持不变。
reduction：控制损失的降维方式。可选值为 'none'、'sum' 或 'mean'，分别表示不降维、求和和取均值。'''
true_lable=[]
opt=torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    opt.zero_grad() #防止之前的梯度积累
    outputs=model(input_data)
    loss=Loss(outputs,true_lable)
    loss.backward() #计算得到梯度 信息被存储在.grad中
    opt.step() #根据梯度，优化器，进行参数更新

    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    '''f'...' 是一种称为 f-string 的字符串格式化方式，它允许您在字符串中插入变量，并在字符串中使用大括号 {} 来表示变量的值。loss.item() 用于获取损失张量的标量值（即 Python 的浮点数），而 :.4f 表示输出这个浮点数保留四位小数。'''

#model、loss、optimizer

#model.train(), model.eval()
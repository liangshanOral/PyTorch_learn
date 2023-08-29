import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

import datetime #用来处理时间和日期
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #mpl下的一个子模块用来绘制3d图形
from pylab import mpl #mpl下子模块，有很多绘图计算函数

#hyper parameter
num_time_steps=16
input_size=3
hidden_size=16
output_size=3
num_layers=1
LR=0.01

#model
class rnn(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(rnn,self).__init__()
        self.rnn=nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, #让batch变成第一个维度
        )
        for p in self.parameters():
            nn.init.normal_(p,mean=0.0,std=0.001) #将参数初始化

        self.linear=nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden_prev): #hidden_prev就是隐藏层初始状态
        out,hidden_prev=self.rnn(x,hidden_prev)
        out=out.view(-1,hidden_size) #view用来自动转换维度，-1表示自动计算
        out=self.linear(out)
        out=out.unsqueeze(dim=0)
        return out,hidden_prev

#dataset
def getdata():
    x1=np.linspace(1,10,30).reshape(30,1) #reshape将一维数据变成二维
    y1=(np.zeros_like(x1)+2)+np.random.rand(30,1)*0.1
    z1=(np.zeros_like(x1)+2).reshape(30,1)

    tr1=np.concatenate((x1,y1,z1),axis=1)
    return tr1 #乱生成的轨迹数据啦

def train_rnn(data):
    model=rnn(input_size,hidden_size,num_layers)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),LR)

    hidden_prev=torch.zeros(1,1,hidden_size) #初始batch、seq_len为1
    l=[]

    model.train()
    for iter in range(200):
        start=np.random.randint(10,size=1)[0] #生成0-9之间的一个数
        end=start+15
        x=torch.tensor(data[start:end]).float().view(1,num_time_steps-1,3)
        y=torch.tensor(data[start+5:end+5]).float().view(1,num_time_steps-1,3) #这里加几根本不重要，因为模型还没有被训练，+5说明用来5步以后的数据用来预测训练呗

        output,hidden_prev=model(x,hidden_prev)
        hidden_prev=hidden_prev.detach() #把这个张量分离出来画图用

        loss=criterion(output,y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter%10==0:
            print("Iteration:{} loss {}".format(iter,loss.item()))
            l.append(loss.item())

    plt.plot(l,'r')
    plt.xlabel('训练次数')
    plt.ylabel('loss')
    plt.title('RNN损失函数下降曲线')

    return hidden_prev,model

def eval_rnn(model,data,hidden_prev):
    data_test=data[19:29]
    data_test=torch._reshape_from_tensor(np.expand_dims(data_test,axis=0),dtype=torch.float32)

    model.eval()

    pred1,h1 = model(data_test,hidden_prev)
    print('pred1.shape:',pred1.shape)
    pred2,h2 = model(pred1,hidden_prev )
    print('pred2.shape:',pred2.shape)
    pred1 = pred1.detach().numpy().reshape(10,3)
    pred2 = pred2.detach().numpy().reshape(10,3)
    predictions = np.concatenate((pred1,pred2),axis=0)
    # predictions= mm.inverse_transform(predictions)
    print('predictions.shape:',predictions.shape)

    #预测可视化

    fig=plt.figure(figsize=(9,6))
    ax=Axes3D(fig)
    ax.scatter3D(data[:,0],data[:,1],data[:,2],c='red')
    ax.scatter3D(predictions[:,0],predictions[:,1],predictions[:,2],c='y')
    ax.set_xlabel('X')
    ax.set_xlim(0, 8.5)
    ax.set_ylabel('Y')
    ax.set_ylim(0, 10)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 4)
    plt.title("RNN航迹预测")
    plt.show()

def main():
    data=getdata()
    start=datetime.datetime.now()
    hidden_prev,model=train_rnn(data)
    end=datetime.datetime.now()
    print('the training time %s'% str(end-start))
    plt.show()
    #eval_rnn(model,data,hidden_prev)

if __name__=='__main__':
    main()


#openmp出错不知道什么问题，明天再说
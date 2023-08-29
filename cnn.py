import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision #这个里面包含了一些常用数据集啥的

torch.manual_seed(250) #从同样的种子开始随机，保证结果复现

#hyper parameters
EPOCH=2
LR=0.01
BATCH_SIZE=50
DOWNLOAD_MNIST=True

#datasets
train_data=torchvision.datasets.MNIST(
    root='./MNIST/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_data=torchvision.datasets.MNIST(root='./MNIST/',train=False)

train_loader=Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_x=torch.unsqueeze(test_data.data,dim=1).float()[:2000]/255.0
#unsqueeze是用来增加维度的，dim指定在哪一维增加维度，源数据是B，H，W的灰度图像，没有通道数，所以加个C=1
test_y=test_data.targets[:2000]

#model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1, # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 
            padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out=nn.Linear(32*7*7,10)

    def forward(self,x): #其他都是self，x代表数据
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) #batch_size,32*7*7
        out=self.out(x)

        return out

model=CNN()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
loss=nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        pred_y=model(batch_x)
        train_loss=loss(pred_y,batch_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if step % 50 ==0:
            test_output=model(test_x)
            pred_t=torch.max(test_output,1)[1].numpy()

            print('Epoch:',epoch,'|train loss:%.4f'%train_loss.data.numpy())

model.eval()
test_output=model(test_x[:10])
pred_y=torch.max(test_output,1)[1].numpy()
print(pred_y,'predicted number')
print(test_y[:10].numpy(),'true number')

#over!
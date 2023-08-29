
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

'''
#pytorch中数据集导入
training_data=datasets.FashionMNIST(
    root="data", #下载后数据集放在哪个路径
    train=True, #说明是训练集
    download=True,
    transform=ToTensor()
)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
'''
'''
print(training_data[0][0].shape)
img=training_data[0][0]
plt.imshow(img.squeeze()) #squeeze删除所有维度为1的维度，而且imshow只能处理二维数据，所以不加的时候就会报错
plt.show()
'''

#print(training_data[5][0]) #可以看到图片已经变成了tensor，输出的是每个像素的值
# Q:第二个索引代表什么 0-图片 1-图片所属类
#print(training_data[5][1]) 得到图片标签

'''
#利用自带函数快速构建数据集-指定该函数的根目录可以快速根据次级目录及其中图片构建数据集
training_set=datasets.ImageFolder(root='training-set')  #次级目录很重要
for i in range(len(training_set)):
    print(i,training_set[i])
print(training_set.classes) #这就是次级目录的作用，代表名称
print(training_set.class_to_idx) #类别索引
print(training_set.imgs) #所有信息
ref:
#https://blog.csdn.net/taylorman/article/details/118631209
'''
'''
#从官方数据集载入也可以看出，构建数据集需要对应的图片（tensor）+标签（target），所以可以用字典+列表保存所有
#但这个只是个人尝试，只能单进程，其实效果不稳健
from torchvision.io import read_image
import os

def load_data():
    training_root=''
    test_root=''

    targets=["class1","class2"]

    training_set={"flower":[],"bird":[]}
    test_set={"flower":[],"bird":[]}

    for root,dirs,files in os.walk(training_root): #root 文件夹路径， dirs 文件夹名称
        target=root.split('\\')[-1] #也就是说预先定义了类，然后图片文件夹也需要是这个类，才能进行匹配
        if target in targets:
            print(root,target)
            for file in files:
                pic=read_image(os.path.join(root,file))
                training_set[target].append(pic)

    #test集同理
'''
#best customize way-继承Dataset类并重载相关方法 
#1 map _getitem_ _len_ 2 iter 覆写_item_ 但不是很清楚具体操作，for tomorrow 8.7

'''
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch
import os

class self_dataset(Dataset):
    def __init__(self,root,target)->None: #None表示返回的是空值，即不返回任何东西
        super().__init__() #super() 用来调用基类的方法，此句表明调用父类的init方法
        self.root=root
        self.target=target
        self.set = []
        self.transform=transforms.Compose([
                transforms.Resize((200,200)),
                transforms.CenterCrop((100,100)),
                transforms.ConvertImageDtype(torch.float64),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]), 
            ])
        
        for root,dirs,files in os.walk(self.root): #一次性把数据全部读到内存中，会受限于内存大小
            classes=root.split('\\')[-1]
            if classes in self.target:
                for file in files:
                    pic=read_image(os.path.join(root,file))
                    pic=self.transform(pic) #无语了
                    information={
                        'image':pic,
                        'target':classes
                    }
                    self.set.append(information)

    def __getitem__(self,index):
        return self.set[index]
    
    def __len__(self):
        return len(self.set)

def load_datasets():
  
    training_root="training-set"
    test_root=""

    targets=["photos","road"]

    training_set=self_dataset(root=training_root,target=targets)
    #test_set

    print(len(training_set))
    
    training_dataloader=DataLoader(training_set,batch_size=2)
    for idx, data in enumerate(training_dataloader): #dataloader无item属性但可以通过迭代方式获取图片信息
       print(idx,data['image'].shape,data['target'])
       print(data['image'].max(),data['image'].min())
       
       
load_datasets()
'''
#iteration版

#dataloader可以看成是有了数据集之后进行的二次操作，将数据转换成要训练的样子

#io读取浪费时间，所以换一个更高效的方式，用h5py

import h5py
import os
from torchvision import transforms
from torchvision.io import read_image
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

def create_file():
    training_root="training-set"
    test_root=""

    targets=["class1","class2"]

    train_file=h5py.File("train.hdf5","w")
    test_file=h5py.File("test.hdf5","w")

    transform=transforms.Compose([
                transforms.Resize((200,200)),
                transforms.CenterCrop((100,100)),
                transforms.ConvertImageDtype(torch.float64),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]), #直接给出均值及方差
                '''# 创建一个批归一化层
                    batch_norm = nn.BatchNorm2d(num_features=input_features)

                    # 创建一个随机输入张量，模拟一个小批次的图像数据
                    input_tensor = torch.randn(10, input_features, image_size, image_size)

                    # 在输入张量上应用批归一化
                    output_tensor = batch_norm(input_tensor)
                    this way '''
            ])
        
    train_data=[]
    train_target=[]
    test_data=[]
    test_target=[]

    for root,dirs,files in os.walk(training_root):
        target=root.split('\\')[-1]
        if target in targets:
            for file in files:
                pic=read_image(os.path.join(root,file))
                pic=transforms(pic)
                pic = np.array(pic).astype(np.float64)

                train_data.append(pic)
                train_target.append(target)

    train_file.create_dataset("image",data=train_data)
    train_file.create_dataset("dataset",data=train_target)

    train_file.close()
    #...

    training_set={"flower":[],"bird":[]}
    test_set={"flower":[],"bird":[]}

class h5py_dataset(Dataset):
    
    def __init__(self,file_name) -> None:
        super().__init__()
        self.file_name = file_name
        
    def __getitem__(self, index):
        with h5py.File(self.file_name,'r') as f:
            if f['target'][index].decode() == "bird":
                target = torch.tensor(0)
            else :
                target = torch.tensor(1)
            return f['image'][index] , target
        
    def __len__(self):
        with h5py.File(self.file_name,'r') as f:
            return len(f['image'])
        
create_file()
h5=h5py_dataset(file_name="train.hdf5")
print(h5.__len__)

#如果图片太小可以采取图片增强等形式

#总之，尝试了使用官方自带数据集、快速读取、自己写、重载DataSet（两种数据集形式），进行transform等操作，在dataloader中进行batch等的定义
#ref：https://zhuanlan.zhihu.com/p/466699075
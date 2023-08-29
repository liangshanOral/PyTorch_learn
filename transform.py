'''
常见方式
将图片转换成tensor算子
ps:utils是常用的工具箱集合
'''
import distutils
from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter

img_path="training-set/photos/OIP-C.jpg" #还是记不住/
img_PIL=Image.open(img_path)
img_ndarray=cv2.imread(img_path)
#采用两种方式读入数据

tensor_tran=transforms.ToTensor() #怎么的totensor还是一个类吗
tensor_img1=tensor_tran(img_PIL)
tensor_img2=tensor_tran(img_ndarray)
#都可以转成tensor

#利用tensorboard进行展示
writer=SummaryWriter("logs_2",comment="add some comment") #give a specific folder name
#通过add_scalar对图表进行操作，定义标题，xy轴什么的
writer.add_image("tensor_img-tag",tensor_img2) 

writer.close()

#normalize 略过
#resize、randomcrop compose filp rotation pad colorjitter grayscale 
#lineartransformation randomaffine 
#lambda
#randomapply randomorder
#反正就这些操作，用的时候再说 https://zhuanlan.zhihu.com/p/53367135
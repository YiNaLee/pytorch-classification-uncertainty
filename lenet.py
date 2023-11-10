import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)#这里的20000是手动计算的，因为对于MNIST问题，28*28为原始输入
        #经过卷积一层：（28-5+2*0）/1+1=24
        #从28*28变成了 24*24
        #然后经过最大池化 kernel size =1 so still 24*24
        #然后第二层卷积： （24-5+2*0）/1+1=20
        #所以最后的特征图是20*20
        #然后通道数是20 所以特征图的总尺寸是20*20*50=20000
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))#池化的kernel size =1没有意义，不改变feature map size

        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)#在卷积层处理完成后，特征是20*20*50的3d格式，需要展平成1维
        x = F.relu(self.fc1(x))#.view函数用于展平特征，送入全连接层
        #.view和.reshape的区别:.view对于非连续张量需要处理，而.reshape()可以自动处理非连续张量
        #reshape被人为更加安全
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

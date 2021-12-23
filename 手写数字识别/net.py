# -*- coding: utf-8 -*-
# @Time : 2021/9/13 11:07
# @Author : 黄小渣
# @FileName: net.py
# @Software: PyCharm
'''
dataset:Mnist 1*28*28  类别：10，对输出的结果进行激活(0,1)
导入数据集
构建网络
train
test
pytorch
'''
import torch
# import torch.nn as nn
from torch import nn
#全连接
class Net_v1(nn.Module):
    #初始化模型
    def __init__(self):
        super(Net_v1, self).__init__()
        # TO DO
        self.fc_layers = nn.Sequential(      # 序列构造器
            nn.Linear(1*28*28,100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Softmax(dim=1)  #列
        )

    def forward(self,x):
        return self.fc_layers(x)

#卷积网络
class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,16,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),# 一般是2/3
            nn.Conv2d(16, 32,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 44,(3,3)),
            nn.ReLU(),
            nn.Conv2d(44, 64,(3,3)),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(64*1*1,10)
        )
    def forward(self,x):
        out = self.layers(x).reshape(-1,64*1*1)
        out = self.out(out)
        return out

if __name__ == "__main__":
    # 全连接网络定义
    # net = Net_v1()
    # input = torch.randn(1,1*28*28)
    # output = net(input)
    # print(output.shape)
    #卷积网络定义
    net = Net_v2()
    x = torch.randn(1,1,28,28)
    print(net(x).shape)

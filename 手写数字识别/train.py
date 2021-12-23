# -*- coding: utf-8 -*-
# @Time : 2021/9/13 11:37
# @Author : 黄小渣
# @FileName: train.py
# @Software: PyCharm
import torch
from  torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter    # loss
from net import Net_v1,Net_v2
import os
from torch import nn , optim
from torch.nn.functional import one_hot
import datetime
'''
Mnist is 60000*1*28*28的数据集
'''

train_dataset = datasets.MNIST('/Users/yiguo/Desktop/项目集合/Bzhan_shenduxuexi_milu/手写数字识别/data',train=True,transform=transforms.ToTensor(),download=True) # train_dataset.data  train_dataset.targets
test_dataset  = datasets.MNIST('/Users/yiguo/Desktop/项目集合/Bzhan_shenduxuexi_milu/手写数字识别/data',train=False,transform=transforms.ToTensor(),download=True)

train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=100,shuffle=True)

# DEVICE = 'cuda'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train_V1:
    def __init__(self,weight_path):
        self.summaryWriter = SummaryWriter('logs')
        self.net = Net_v1().to(device)
        if os.path.exists('weight_path'):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = optim.Adam(self.net.parameters())
        self.fc_loss = nn.MSELoss()
        self.train = True
        self.test = True
    def __call__(self): # 训练的过程
        index1 ,index2 = 0,0
        for epoch in range(100):
            if self.train:
                for i , (img , label) in enumerate(train_dataloader):
                    # 对label进行onehot编码
                    label = one_hot(label,10).float().to(device)
                    img = img.reshape(-1,1*28*28).to(device)
                    train_y = self.net(img)
                    train_loss = self.fc_loss(train_y,label)

                    #清空梯度
                    self.opt.zero_grad()
                    #梯度计算
                    train_loss.backward()
                    #梯度更新
                    self.opt.step()

                    #打印
                    if i%10:
                        print(f"train_loss {i}====>",train_loss.item())
                        #可视化
                        self.summaryWriter.add_scalar('train_loss',train_loss,index1)
                        index1 += 1
                    #每一个批次保存模型
                data_time = str(datetime.datetime.now()).replace(' ','-').replace(':','-').replace('.','-')
                torch.save(self.net.state_dict(),f'param/{data_time}--{epoch}.pt')

            if self.test:
                for i, (img, label) in enumerate(test_dataloader):
                    # 对label进行onehot编码
                    label = one_hot(label, 10).float().to(device)
                    img = img.reshape(-1, 1 * 28 * 28).to(device)
                    test_y = self.net(img)
                    test_loss = self.fc_loss(test_y, label)

                    test_y = torch.argmax(test_y,dim=1)
                    label = torch.argmax(label,dim=1)
                    acc = torch.mean(torch.eq(test_y,label).float())

                    '''
                    # 清空梯度
                    self.opt.zero_grad()
                    # 梯度计算
                    train_loss.backward()
                    # 梯度更新
                    self.opt.step()
                    '''
                    # 打印
                    if i % 10:
                        print(f"train_loss {i}====>", test_loss.item())
                        print(f'acc {i}=====>',acc.item())
                        # 可视化
                        self.summaryWriter.add_scalar('test_loss', test_loss, index1)
                        index2 += 1

class Train_V2:
    def __init__(self,weight_path):
        self.summaryWriter = SummaryWriter('logs')
        self.net = Net_v2().to(device)
        if os.path.exists('weight_path'):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = optim.Adam(self.net.parameters())
        self.fc_loss = nn.MSELoss()
        self.train = True
        self.test = True
    def __call__(self): # 训练的过程
        index1 ,index2 = 0,0
        for epoch in range(1000):
            if self.train:
                for i , (img , label) in enumerate(train_dataloader):
                    # 对label进行onehot编码
                    label = one_hot(label,10).float().to(device)
                    # img = img.reshape(-1,1*28*28).to(device)
                    train_y = self.net(img)
                    train_loss = self.fc_loss(train_y,label)

                    #清空梯度
                    self.opt.zero_grad()
                    #梯度计算
                    train_loss.backward()
                    #梯度更新
                    self.opt.step()

                    #打印
                    if i%10:
                        print(f"train_loss {i}====>",train_loss.item())
                        #可视化
                        self.summaryWriter.add_scalar('train_loss',train_loss,index1)
                        index1 += 1
                    #每一个批次保存模型
                data_time = str(datetime.datetime.now()).replace(' ','-').replace(':','-').replace('.','-')
                torch.save(self.net.state_dict(),f'param/{data_time}--{epoch}.pt')

            if self.test:
                for i, (img, label) in enumerate(test_dataloader):
                    # 对label进行onehot编码
                    label = one_hot(label, 10).float().to(device)
                    # img = img.reshape(-1, 1 * 28 * 28).to(device)
                    test_y = self.net(img)
                    test_loss = self.fc_loss(test_y, label)

                    test_y = torch.argmax(test_y,dim=1)
                    label = torch.argmax(label,dim=1)
                    acc = torch.mean(torch.eq(test_y,label).float())

                    '''
                    # 清空梯度
                    self.opt.zero_grad()
                    # 梯度计算
                    train_loss.backward()
                    # 梯度更新
                    self.opt.step()
                    '''
                    # 打印
                    if i % 10:
                        print(f"train_loss {i}====>", test_loss.item())
                        print(f'acc {i}=====>',acc.item())
                        # 可视化
                        self.summaryWriter.add_scalar('test_loss', test_loss, index1)
                        index2 += 1


if __name__ == '__main__':
    train = Train_V2('./param1/1.pt')
    train()
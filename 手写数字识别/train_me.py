import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter    # loss
import torch.nn as nn
import datetime
import os
from net import Net_v1,Net_v2
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
train_dataset = datasets.MNIST('/Users/yiguo/Desktop/项目集合/Bzhan_shenduxuexi_milu/手写数字识别/data',transform = transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST('/Users/yiguo/Desktop/项目集合/Bzhan_shenduxuexi_milu/手写数字识别/data',transform = transforms.ToTensor(),download=True)

train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=100,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

#数据加载完成，我们需要进行训练
class TrainV1:
    def __init__(self,weight_path):
        '''理解成这部分是一些组件，方便以后搭建'''
        self.summaryWrittet = SummaryWriter("logs")
        self.net =  Net_v1().to(device)
        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.optim = optim.Adam(self.net.parameters())
        self.fc_loss = nn.MSELoss()
        self.train = True
        self.test = True

    def __call__(self):
        for epoch in range(10):
            if self.train:
                for i, (img, label) in enumerate(train_dataloader):
                    label = one_hot(label,10).float().to(device)
                    #我们对图像进行操作 torch.Size([100, 1, 28, 28])变成(batchsize,1*28*28)
                    img = img.reshape(-1,1*28*28).to(device)
                    train_y = self.net(img)
                    train_loss = self.fc_loss(train_y,label)
                    #清空梯度
                    self.optim.zero_grad()
                    #梯度计算
                    train_loss.backward()
                    #梯度更新
                    self.optim.step()

                    #打印
                    if i%10:
                        print(f"train_loss {i}====>",train_loss.item())

                #每一个批次保存模型
                data_time = str(datetime.datetime.now()).replace(' ','-').replace(':','-').replace('.','-')
                torch.save(self.net.state_dict(),f'param/{data_time}--{epoch}.pt')

            if self.test:   #你其实可以加上
                with torch.no_grad():
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
                            print(f"test_loss {i}====>", test_loss.item())
                            print(f'acc {i}=====>',acc.item())

if __name__ == '__main__':
    train = TrainV1('./param1/1.pt')
    train()
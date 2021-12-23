import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from net import Net_v1,Net_v2
import os
from torch.nn.functional import one_hot

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train_v1:
    def __init__(self,weight_path):
        self.net = Net_v1().to(device)
        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.optim = optim.Adam(self.net.parameters())
        self.loss = nn.MSELoss()
        self.train = True
        self.test = True
    def __call__(self):
        for epoch in range(5):
            if self.train:
                for i, (img, label) in enumerate(train_dataloader):
                    label = one_hot(label,10).float().to(device)
                    img = img.reshape(-1,1*28*28).to(device)
                    y_train = self.net(img)
                    train_loss = self.loss(y_train, label)

                    self.optim.zero_grad()
                    train_loss.backward()
                    self.optim.step()

                    if i%10:
                        print('{}====> train_loss is {}'.format(i, train_loss))
            torch.save(self.net.state_dict(),f"./param/best.pt")

            if self.test:
                with torch.no_grad():
                    for i, (img,label) in enumerate(test_dataloader):
                        label = one_hot(label,10).float().to(device)
                        img = img.reshape(-1, 1*28*28).to(device)
                        y_test = self.net(img)
                        test_loss = self.loss(y_test,label)
                        y_test = torch.argmax(y_test, dim=1)
                        label = torch.argmax(label, dim=1)
                        acc = torch.mean(torch.eq(y_test, label).float())
                        print(acc.item())

if __name__ == '__main__':
    net_v1 = Train_v1("./param/v1.pt")
    net_v1()




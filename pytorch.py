#!/usr/bin/env python
# coding: utf-8

# # Pytorch Tutorial

# Pytorch is a popular deep learning framework and it's easy to get started.

# In[1]:


import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import torch.nn.functional as F
BATCH_SIZE=128 #大概需要2G的显存
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
# DEVICE = torch.device("cpu")
# First, we read the mnist data, preprocess them and encapsulate them into dataloader form.

# In[2]:


# preprocessing
# train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('mnist', train=True, download=True, 
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=BATCH_SIZE, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('mnist', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=BATCH_SIZE, shuffle=True)
# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
# Then, we define the model, object function and optimizer that we use to classify.

# In[12]:

class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.fc3= nn.Sequential(
            nn.Linear(256, 10),
            nn.Dropout(),
            nn.LogSoftmax(dim=1)
        )
    def forward(self,x):
        x = x.view(BATCH_SIZE, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Sequential(
            nn.Conv2d(1,10,5), # 10, 24x24
            # nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(10,20,3), # 128, 10x10
            # nn.BatchNorm2d(20),
            nn.ReLU(True),
        ) 
        self.fc1 = nn.Sequential(
            nn.Linear(20*10*10,128),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,10),
            nn.LogSoftmax(dim=1),
        )
    def forward(self,x):
        in_size = x.size(0)
        
        out = self.conv1(x) #24
        
        out = self.conv2(out) #10
        out = out.view(in_size,-1)
        
        out = self.fc1(out)
        
        out = self.fc2(out)
        return out


model = ConvNet()
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)

# 定义损失函数和优化器
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())


# Next, we can start to train and evaluate!

# In[14]:
# DEVICE = torch.device("cpu")
model = model.to(DEVICE)
# train and evaluate
# 训练模型
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        output = model(data)
#         print(output.shape, target.shape)
#         print(output.dtype, target.dtype)
#         print('\n\n')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
#        if(batch_idx+1)%30 == 0: 
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader), loss))

    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset

def test_training(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)# 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(train_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
#
def test_testing(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)# 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#
 #### Q5:
# Please print the training and testing accuracy.
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test_training(model, DEVICE, train_loader)
    test_testing(model, DEVICE, test_loader)

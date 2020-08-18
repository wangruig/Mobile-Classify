#coding=utf-8
from __future__ import print_function  
import torch
import numpy as np
import pandas as pd
import time 
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import torch.utils.data as Data
# start_time=time.time()

path=r'C:\Users\wangrg\Desktop\kaggle_ML'
train=pd.read_csv(path+r'\\Mobile price\train.csv')
train_data=train.iloc[:,:20]
train_target=train.iloc[:,20]

#网络模型

class Net(torch.nn.Module):
    # input_dims = 20
    def __init__(self):
        super(Net,self).__init__()
        self.net = torch.nn.Sequential(         
            torch.nn.Linear(20,15)
            # ,torch.nn.BatchNormld(10)
            ,torch.nn.Sigmoid()
            ,torch.nn.Dropout(0.6)
            ,torch.nn.Linear(15,10)
            # ,torch.nn.BatchNormld(10)
            ,torch.nn.Sigmoid()
            ,torch.nn.Dropout(0.6)
            ,torch.nn.Linear(10,4)
            )
        
        self.net1 = torch.nn.Softmax(2)
    def forward(self,x):
        out = self.net(x)
        out = F.softmax(out, dim=1) # 计算log(softmax(x))
        return out


def get_data(batch_size=256):
    x_train,x_test,y_train,y_test = train_test_split(train_data,train_target,test_size=0.1,random_state=42)

    x_train = x_train.values 
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train) 
    x_test = torch.from_numpy(x_test) 
    y_test = torch.from_numpy(y_test)

    train_dataset = Data.TensorDataset(x_train, y_train) 
    test_dataset = Data.TensorDataset(x_test, y_test)  
    train_dataset = Data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = Data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset, test_dataset 


def train(model,train_dataset,test_dataset):
    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optm = torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9, 0.99))
    epochs = 1500
    train_loss = []
    valid_loss = []
    for j in range(epochs):
        net.train()
        for i, data in enumerate(train_dataset):
            optm.zero_grad()
            (inputs, labels) = data
            inputs = inputs.to(device, dtype=torch.float)  # 100*20
            labels = labels.to(device, dtype=torch.float)  # 100*1
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)
            outputs = net(inputs)
            # outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optm.step()
            train_loss.append(loss.item())
        net.eval()
        for i, test_data in enumerate(test_dataset):
            (inputs, labels) = test_data
            inputs = inputs.to(device, dtype=torch.float)  # 100*20
            labels = labels.to(device, dtype=torch.float)  # 100*1
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)
            outputs = net(inputs)
            # outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels.long())
            valid_loss.append(loss.item())
        print('epcoch %s  train loss: %.6f, valid loss:  %.6f' % (j, np.average(train_loss), np.average(valid_loss)))
        train_loss = []
        valid_loss = []
        if j % 5 == 0:
            test(net, test_dataset)
    # save model 
    # torch.save(net.state_dict(), current_path + '/../model/fcnet_train.pth')

    return net


def test(model, test_dataset):
    # start evaling
    model.eval()
    y_test_list = []
    y_pred_list = []
    correct = 0
    with torch.no_grad():
        for (x_test, y_test) in test_dataset:
            x_test = x_test.to(device, dtype=torch.float)
            y_test = y_test.to(device, dtype=torch.float)
            y_pred = model(x_test)
            y_pred = y_pred.max(1, keepdim=True)[1] # 找到概率最大的下标
            y_test_list.extend(y_test.cpu().data.numpy().tolist())
            y_pred_list.extend(y_pred.reshape(-1).cpu().data.numpy().tolist())
            correct += y_pred.eq(y_test.view_as(y_pred)).sum().item()
    print('Accurary:{} {} {:.2f}'.format(correct, len(test_dataset.dataset), correct/len(test_dataset.dataset)))

if __name__ =='__main__':
    # 加载数据
    train_dataset, test_dataset = get_data(100)
    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()

    # start training
    net = train(net, train_dataset, test_dataset)

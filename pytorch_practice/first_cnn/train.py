# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-26 14:08:36
# @Last Modified by:   None
# @Last Modified time: 2022-01-09 19:54:29
import torch
import torchvision
from torch import nn
from torch.utils import data as Data
from model import LeNet
from show_process import Shower


model = LeNet()
epoch = 5
batch_size = 64
lr = 0.001


transform = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST('./data', train=True, download=False,
                                        transform=transform)
train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                               drop_last=True)
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)
torch.set_grad_enabled(True)
model.train()
device = torch.device("cude:0" if torch.cuda.is_available() else 'cpu')
model.to(device)

# shower = Shower(env='train', win='train')
for epoch_index in range(epoch):
    running_loss = 0.0
    accuracy = 0.0
    for step, (datas, lables) in enumerate(train_loader):
        opt.zero_grad()
        y_pre = model(datas.to(device, torch.float))
        pre = y_pre.argmax(axis=1)
        accuracy += (pre.detach().cpu() == lables.detach()).sum()
        loss = loss_func(y_pre, lables)
        running_loss += float(loss.detach().cpu())
        loss.backward()
        opt.step()
        if step % 100 == 99:
            average_loss = running_loss / (step + 1)
            acc = float(accuracy / ((step + 1) * batch_size))
            print(f'Epoch:{epoch_index + 1}', f'step:{step + 1}',
                  f'average loss:{average_loss:.6f}', f'accuracy:{acc:.6f}', sep='\n')
            # shower(torch.tensor([[average_loss, acc]]))
torch.save(model, './LeNet.pkl')

# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-27 16:58:45
# @Last Modified by:   None
# @Last Modified time: 2022-01-09 14:12:17
import tqdm
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from model import ResNet34
from dataset import CatDogSet
from show_process import Shower

net = ResNet34()
net.init_weight()
epoch = 10
batch_size = 32
lr = 0.001
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.5)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()])
train_data = CatDogSet('./dataset', True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True,
                          shuffle=True)
validation_data = CatDogSet('./dataset/validation.zip', transform=transform)
validation_loader = DataLoader(validation_data, batch_size=batch_size,
                               shuffle=True, drop_last=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)


def fit(model, loader, train, shower):
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.
    accuracy = 0.
    for step, (datas, lables) in enumerate(loader):
        y_pre = model(datas.to(device, torch.float))
        pre = y_pre.argmax(axis=1)
        accuracy += (pre.detach().cpu() == lables.detach()).sum()
        loss = loss_func(y_pre, lables)
        running_loss += loss.detach().cpu()
        if train:
            loss.backward()
            opt.step()
            opt.zero_grad()
        if step % 100 == 99:
            ave_loss = running_loss / (step + 1)
            ave_acc = accuracy / ((step + 1) * batch_size)
            shower(np.array([[ave_loss, ave_acc]]))
    if train:
        scheduler.step()
    return running_loss / (step + 1), accuracy / ((step + 1) * batch_size)


def train():
    train_shower = Shower(env='table', win='train')
    validation_shower = Shower(env='table', win='validation')
    for epoch_index in range(epoch):
        train_loss, train_acc = fit(net, train_loader, True, train_shower)
        val_loss, val_acc = fit(net, validation_loader, False, validation_shower)
        print(f'Epoch:{epoch_index + 1}',
              f'train loss:{train_loss}\tvalidation loss:{val_loss}',
              f'train accuracy:{train_acc}\tvalidation accuracy:{val_acc}')
    torch.save(net.state_dict(), 'ResNet.pth')


if __name__ == '__main__':
    train()

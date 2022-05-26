# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-28 15:26:07
# @Last Modified by:   None
# @Last Modified time: 2021-12-28 15:38:41
import torch
import torchvision
from torch.utils.data import DataLoader
from model import ResNet34
from dataset import CatDogSet


net = ResNet34()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('./ResNet34.pth', map_location=torch.device(device))
net.load_state_dict(state_dict)
net.to(device)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()])
test_data = CatDogSet('./dataset', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=1)
torch.set_grad_enabled(False)
net.eval()
for data, lable in test_loader:
    y_pre = net(data.to(device, torch.float))
    pre = y_pre.argmax(axis=1)

# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-26 16:47:15
# @Last Modified by:   None
# @Last Modified time: 2021-12-26 20:34:26
import torch
import torchvision
from torch.utils import data as Data


transform = torchvision.transforms.ToTensor()
test_data = torchvision.datasets.MNIST('./data', train=False, download=False,
                                       transform=transform)
test_loader = Data.DataLoader(test_data, shuffle=False, batch_size=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = torch.load('./LeNet.pkl', map_location=torch.device(device))
net.to(device)
torch.set_grad_enabled(False)
net.eval()
length = test_data.data.size(0)
accuracy = 0.0
for datas, lables in test_loader:
    y_pre = net(datas.to(device, torch.float))
    pre = y_pre.argmax(axis=1)
    print(f'real:{lables}\t\tpredict:{pre}')
    accuracy += (pre.detach().cpu() == lables.detach()).sum()
print(f'accuracy:{accuracy / length:.6f}')

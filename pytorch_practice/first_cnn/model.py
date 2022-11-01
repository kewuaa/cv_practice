# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-26 09:06:51
# @Last Modified by:   None
# @Last Modified time: 2022-01-09 20:58:05
import torch
from torch import nn


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, 5), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.layer5 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        out = self.layer5(x)
        return out


if __name__ == '__main__':
    model = LeNet()
    x = torch.randn(28, 1, 28, 28)
    y = model(x)
    print(y)

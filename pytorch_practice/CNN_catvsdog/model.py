# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-26 21:59:08
# @Last Modified by:   None
# @Last Modified time: 2021-12-27 22:08:37
import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(x)
        out = out + x
        out = F.relu(out)
        return out


class ResBlockDown(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlockDown, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.pool = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, x):
        res = self.pool(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out + res
        out = F.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer2 = nn.Sequential(
            ResBlockDown(64, 64), ResBlock(64, 64), ResBlock(64, 64))
        self.layer3 = nn.Sequential(
            ResBlockDown(64, 128), ResBlock(128, 128), ResBlock(128, 128),
            ResBlock(128, 128))
        self.layer4 = nn.Sequential(
            ResBlockDown(128, 256), ResBlock(256, 256), ResBlock(256, 256),
            ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 256))
        self.layer5 = nn.Sequential(
            ResBlockDown(256, 512), ResBlock(512, 512))
        self.gra = nn.AdaptiveAvgPool2d(1)
        self.outlayer = nn.Linear(512, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gra(out)
        out = out.view(out.size(0), -1)
        out = self.outlayer(out)
        return out

    def init_weight(self):
        for moudle in self.modules():
            if isinstance(moudle, nn.Conv2d):
                nn.init.xavier_normal_(moudle.weight)
            elif isinstance(moudle, nn.BatchNorm2d):
                nn.init.constant_(moudle.weight, 1.)
                nn.init.constant_(moudle.bias, 0.)


if __name__ == '__main__':
    model = ResNet34()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())

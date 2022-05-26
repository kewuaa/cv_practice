# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-27 12:40:17
# @Last Modified by:   None
# @Last Modified time: 2021-12-27 22:27:21
import os
import zipfile
import torch
import torchvision
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset


def load_zipfile(path):
    with zipfile.ZipFile(path, mode='r') as zf:
        name_list = zf.namelist()
        name_list.pop(0)
        length = len(name_list)
        yield length
        for _ in range(length):
            index = yield
            name = name_list[index]
            with zf.open(name, mode='r') as f:
                pic = BytesIO(f.read())
            yield Image.open(pic).convert('RGB'), name


class CatDogSet(Dataset):

    def __init__(self, folder, train=None, transform=None):
        super(CatDogSet, self).__init__()
        self.folder = folder
        self.train = train
        self.transform = transform
        if self.train is not None:
            self.folder = os.path.join(
                self.folder, f'{"train" if self.train else "test"}.zip')
        self.datas = load_zipfile(self.folder)
        self.length = next(self.datas)

    def __getitem__(self, index):
        next(self.datas)
        img, name = self.datas.send(index)
        if self.train:
            if 'cat' in name:
                lable = 1
            else:
                lable = 0
        else:
            name = os.path.basename(name)
            lable, _ = os.path.splitext(name)
        if self.transform is not None:
            img = self.transform(img)
        return img, lable

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = CatDogSet(
        './dataset/validation.zip', train=None, transform=torchvision.transforms.ToTensor())
    # data, lable = dataset[999]
    print(len(dataset))

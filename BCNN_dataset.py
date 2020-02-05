# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:07:51 2020

@author: ph
"""

import math
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torchvision.models as models

class BCNN(nn.Module):
    def __init__(self):
        super(BCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove avgpool fc.
        self.fc = nn.Linear(512 ** 2, 3)
        

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size()[0], 512, x.size(2) ** 2)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 28 ** 2).view(x.size()[0], -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = torch.sigmoid(self.fc(x))
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
    
class MyDataset(data.Dataset):
    def __init__(self, paths, labels, transform=None, target_transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        path = self.paths[index]        # 图像路径
        image = Image.open(path).convert('RGB')
        # image = image.resize((234, 234), resample=Image.LANCZOS)
        label = self.labels[index]          # 图像标签
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.labels)
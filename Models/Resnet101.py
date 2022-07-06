import torch.nn as nn
from torchvision.models import resnet101
import torch

class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        resnet = resnet101(pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)         
        
        self.linear_1 = nn.Linear(2048, 1024)
        self.bn_1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.linear_2 = nn.Linear(1024, 512)
        self.bn_2 = nn.BatchNorm1d(512, momentum=0.01)
        self.linear_3 = nn.Linear(512, 128)
        self.bn_3 = nn.BatchNorm1d(128, momentum=0.01)
        self.linear_4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, images):
        x = self.resnet(images)
        x = x.reshape(x.size(0), -1)
        x = self.bn_1(self.linear_1(x))
        x = self.bn_2(self.linear_2(x))
        x = self.bn_3(self.linear_3(x))
        x = self.sigmoid(self.linear_4(x))
        
        return x
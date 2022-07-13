import torch.nn as nn
from torchvision.models import densenet121
import torch

class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        modules = densenet121().features
        self.densenet = nn.Sequential(*modules)         
        self.linear_1 = nn.Linear(50176, 4048)
        self.bn_1 = nn.BatchNorm1d(4048, momentum=0.01)
        self.linear_2 = nn.Linear(4048, 1024)
        self.bn_2 = nn.BatchNorm1d(1024, momentum=0.01)
        self.linear_3 = nn.Linear(1024, 512)
        self.bn_3 = nn.BatchNorm1d(512, momentum=0.01)
        self.linear_4 = nn.Linear(512, 128)
        self.bn_4 = nn.BatchNorm1d(128, momentum=0.01)
        self.linear_5 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.densenet(images)
        x = x.reshape(x.size(0), -1)
        x = self.bn_1(self.linear_1(x))
        x = self.bn_2(self.linear_2(x))
        x = self.bn_3(self.linear_3(x))
        x = self.bn_4(self.linear_4(x))
        x = self.sigmoid(self.linear_5(x))
        
        return x
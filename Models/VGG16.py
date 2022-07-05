import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_4_5 = nn.BatchNorm2d(512)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(8)
        self.drop_out = nn.Dropout(0.1)  
        self.linear_1 = nn.Linear(512*8*8, 4096)
        self.linear_2 = nn.Linear(4096, 512)
        self.linear_3 = nn.Linear(512, 64)
        self.linear_4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv1_1(x)))
        x = self.relu(self.bn_1(self.conv1_2(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn_2(self.conv2_1(x)))
        x = self.relu(self.bn_2(self.conv2_2(x)))      
        x = self.max_pool(x)
        x = self.relu(self.bn_3(self.conv3_1(x)))
        x = self.relu(self.bn_3(self.conv3_2(x)))
        x = self.relu(self.bn_3(self.conv3_3(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn_4_5(self.conv4_1(x)))
        x = self.relu(self.bn_4_5(self.conv4_2(x)))
        x = self.relu(self.bn_4_5(self.conv4_3(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn_4_5(self.conv5_1(x)))
        x = self.relu(self.bn_4_5(self.conv5_2(x)))
        x = self.relu(self.bn_4_5(self.conv5_3(x)))
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(self.relu(self.linear_1(x)))
        x = self.drop_out(self.relu(self.linear_2(x)))
        x = self.drop_out(self.relu(self.linear_3(x)))
        x = self.sigmoid(self.linear_4(x))

        return x
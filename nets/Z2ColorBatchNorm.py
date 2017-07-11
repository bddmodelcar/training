import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as initialization
from torch.autograd import Variable


class Z2ColorBatchNorm(nn.Module):
    def __init__(self):
        super(Z2ColorBatchNorm, self).__init__()

        self.lr = 0.1
        self.momentum = 0.1
        self.N_FRAMES = 2
        self.N_STEPS = 10

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=96, kernel_size=11, stride=3, groups=1)
        self.conv1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1_pool_norm = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(in_channels=102, out_channels=256, kernel_size=3, stride=2, groups=2)
        self.conv2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_pool_norm = nn.BatchNorm2d(256)
        self.ip1 = nn.Linear(in_features=2560, out_features=512)
        self.ip1_norm = nn.BatchNorm1d(512)
        self.ip2 = nn.Linear(in_features=512, out_features=20)

        # Initialize weights
        nn.init.normal(self.conv1.weight, std=0.00001)
        nn.init.normal(self.conv2.weight, std=0.1)

        nn.init.xavier_normal(self.ip1.weight)
        nn.init.xavier_normal(self.ip2.weight)

    def forward(self, x, metadata):
        # conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_pool(x)
        x = self.conv1_pool_norm(x)

        # metadata_concat
        x = torch.cat((metadata, x), 1)

        # conv2
        x = self.conv2_pool_norm(self.conv2_pool(F.relu(self.conv2(x))))
        
        x = x.view(-1, 2560)

        # ip1
        x = self.ip1_norm(F.relu(self.ip1(x)))

        # ip2
        x = self.ip2(x)
        
        return x


def unit_test():
    test_net = Z2ColorBatchNorm()
    a = test_net(Variable(torch.randn(5, 12, 94, 168)), Variable(torch.randn(5, 6, 13, 26)))
    print (a)


unit_test()

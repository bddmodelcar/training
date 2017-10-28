"""SqueezeNet 1.1 modified for Aruco markers."""
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.lr = 0.01
        self.momentum = 0.01
        self.N_FRAMES = 2
        self.N_STEPS = 10
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
        )
        self.post_metadata_features = nn.Sequential(
            Fire(256, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        final_conv = nn.Conv2d(512, 66, kernel_size=1)
        self.final_output_Aruco = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=6)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, metadata):
        x = self.pre_metadata_features(x)
        x = torch.cat((x, metadata), 1)
        x = self.post_metadata_features(x)
        x = self.final_output_Aruco(x)
        x = x.view(x.size(0), -1)
        return x


def unit_test():
    test_net = SqueezeNet()
    a = test_net(
        Variable(
            torch.randn(
                5, 12, 94, 168)), Variable(
            torch.randn(
                5, 128, 23, 41)))
    print(a)


unit_test()

"""SqueezeNet 1.1 modified for regression."""
from __future__ import print_function, unicode_literals

import logging
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

logging.basicConfig(filename='training.log', level=logging.DEBUG)


class Nvidia(nn.Module):

    def __init__(self, n_steps=10, n_frames=2):
        super(Nvidia, self).__init__()

        self.n_steps = n_steps
        self.n_frames = n_frames
        self.conv_nets = nn.Sequential(
            nn.Conv2d(3 * 2 * self.n_frames, 24, kernel_size=5, stride=2),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )
        self.fcl = nn.Sequential(
            nn.Linear(768, 100),
            nn.Linear(100, 4 * self.n_steps)
        )

    def forward(self, x, metadata):
        x = self.conv_nets(x)
        x = x.view(x.size(0), -1)
        x = self.fcl(x)
        x = x.view(x.size(0), -1, 4)
        return x

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])


def unit_test():
    test_net = Nvidia(20, 6)
    a = test_net(Variable(torch.randn(5, 36, 94, 168)),
                 Variable(torch.randn(5, 12, 23, 41)))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())


unit_test()

Net = Nvidia

"""SqueezeNet 1.1 modified for regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)

from Parameters import ARGS

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


class Z2Bala(nn.Module):

    def __init__(self):
        super(Z2Bala, self).__init__()

        self.N_STEPS = 10
        self.metadata_size = (11, 20)
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(ARGS.nframes * 6, 96, kernel_size=12, padding=0, stride=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.post_metadata_features = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(102, 256, kernel_size=3, padding=0, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.final_output = nn.Sequential(
            nn.Linear(256 * 3 * 6, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 20),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight.data, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight.data)

    def forward(self, x, metadata):
        x = self.pre_metadata_features(x)
        x = torch.cat((x, metadata), 1)
        x = self.post_metadata_features(x)
        x = x.view(x.size(0), -1)
        x = self.final_output(x)
        return x


def unit_test():
    test_net = Z2Bala()
    a = test_net(Variable(torch.randn(5, 6 * ARGS.nframes, 94, 168)),
                 Variable(torch.randn(5, 6, 14, 26)))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')


unit_test()

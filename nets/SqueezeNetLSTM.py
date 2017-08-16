"""SqueezeNet 1.1 modified for LSTM regression."""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)

#from Parameters import ARGS

class Fire(nn.Module):
    """Implementation of Fire module"""

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        """Sets up layers for Fire module"""
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
        """Forward-propagates data through Fire module"""
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNetLSTM(nn.Module):
    """SqueezeNet+LSTM for end to end autonomous driving"""

    def __init__(self):
        """Sets up layers"""
        super(SqueezeNetLSTM, self).__init__()

        self.N_STEPS = 10
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(2 * 6, 64, kernel_size=3, stride=2),
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
        final_conv = nn.Conv2d(512, self.N_STEPS * 2, kernel_size=1)
        self.pre_lstm_output = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.lstm = nn.LSTM(16, 2, 8, batch_first=True)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                if mod is final_conv:
                    init.normal(mod.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(mod.weight.data)
                if mod.bias is not None:
                    mod.bias.data.zero_()

    def forward(self, x, metadata):
        """Forward-propagates data through SqueezeNetLSTM"""
        x = self.pre_metadata_features(x)
        x = torch.cat((x, metadata), 1)
        x = self.post_metadata_features(x)
        x = self.pre_lstm_output(x)
        x = x.view(x.size(0), self.N_STEPS, -1)
        x = self.lstm(x)[0]
        x = x.contiguous().view(x.size(0), -1)
        return x


def unit_test():
    """Tests SqueezeNetLSTM for size constitency"""
    test_net = SqueezeNetLSTM()
    a = test_net(Variable(torch.randn(5, 2 * 6, 94, 168)),
                 Variable(torch.randn(5, 128, 23, 41)))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')


unit_test()

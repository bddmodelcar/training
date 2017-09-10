"""SqueezeNet 1.1 modified for LSTM regression."""
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

logging.basicConfig(filename='training.log', level=logging.DEBUG)


# from Parameters import ARGS


class Fire(nn.Module):  # pylint: disable=too-few-public-methods
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

    def forward(self, input_data):
        """Forward-propagates data through Fire module"""
        output_data = self.squeeze_activation(self.squeeze(input_data))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(output_data)),
            self.expand3x3_activation(self.expand3x3(output_data))
        ], 1)


class SqueezeNetSqueezeLSTM(nn.Module):  # pylint: disable=too-few-public-methods
    """SqueezeNet+LSTM for end to end autonomous driving"""

    def __init__(self):
        """Sets up layers"""
        super(SqueezeNetSqueezeLSTM, self).__init__()

        self.n_frames = 2
        self.n_steps = 10
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.post_metadata_features = nn.Sequential(
            Fire(148, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        final_conv = nn.Conv2d(512, self.n_steps * 8, kernel_size=1)
        self.pre_lstm_output = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.lstms = nn.ModuleList([
            nn.LSTM(64, 128, 1, batch_first=True),
            nn.LSTM(128, 32, 2, batch_first=True),
            nn.LSTM(32, 64, 1, batch_first=True),
            nn.LSTM(64, 16, 2, batch_first=True),
            nn.LSTM(16, 32, 1, batch_first=True),
            nn.LSTM(32, 8, 2, batch_first=True),
            nn.LSTM(8, 2, 1, batch_first=True)
        ])

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                if mod is final_conv:
                    init.normal(mod.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(mod.weight.data)
                if mod.bias is not None:
                    mod.bias.data.zero_()

    def forward(self, camera_data, metadata):
        """Forward-propagates data through SqueezeNetSqueezeLSTM"""
        net_output = self.pre_metadata_features(camera_data)
        net_output = torch.cat((net_output, metadata), 1)
        net_output = self.post_metadata_features(net_output)
        net_output = self.pre_lstm_output(net_output)
        net_output = net_output.view(net_output.size(0), self.n_steps, -1)
        for lstm in self.lstms:
            net_output = lstm(net_output)[0]
        net_output = net_output.contiguous().view(net_output.size(0), -1)
        return net_output


def unit_test():
    """Tests SqueezeNetSqueezeLSTM for size constitency"""
    test_net = SqueezeNetSqueezeLSTM()
    test_net_output = test_net(
        Variable(
            torch.randn(
                5,
                test_net.n_frames * 6,
                94,
                168)),
        Variable(
            torch.randn(
                5,
                128,
                23,
                41)))
    logging.debug('Net Test Output = {}'.format(test_net_output))
    logging.debug('Network was Unit Tested')


unit_test()

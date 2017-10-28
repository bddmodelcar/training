"""SqueezeNet 1.1 modified for LSTM regression."""
from __future__ import print_function, unicode_literals

import logging
from functools import reduce

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


class SqueezeNetTimeLSTM(nn.Module):  # pylint: disable=too-few-public-methods
    """SqueezeNet+LSTM for end to end autonomous driving"""

    def __init__(self, n_frames=2, n_steps=10):
        """Sets up layers"""
        super(SqueezeNetTimeLSTM, self).__init__()

        self.is_cuda = False

        self.n_frames = n_frames
        self.n_steps = n_steps
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(3 * 2, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(16, 6, 12, 12)
        )
        self.post_metadata_features = nn.Sequential(
            Fire(36, 8, 16, 16),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(32, 12, 24, 24),
            Fire(48, 12, 24, 24),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(48, 16, 32, 32),
            Fire(64, 16, 32, 32),
            Fire(64, 24, 48, 48),
            Fire(96, 24, 48, 48),
        )
        final_conv = nn.Conv2d(96, 2, kernel_size=1)
        self.pre_lstm_output = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.lstm_encoder = nn.ModuleList([
            nn.LSTM(16, 32, 1, batch_first=True)
        ])
        self.lstm_decoder = nn.ModuleList([
            nn.LSTM(1, 32, 1, batch_first=True),
            nn.LSTM(32, 8, 1, batch_first=True),
            nn.LSTM(8, 16, 1, batch_first=True),
            nn.LSTM(16, 4, 1, batch_first=True),
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
        """Forward-propagates data through SqueezeNetTimeLSTM"""
        batch_size = camera_data.size(0)
        nframes = camera_data.size(1) // 6
        metadata = metadata.contiguous().view(-1, 12, 23, 41)
        net_output = camera_data.contiguous().view(-1, 6, 94, 168)
        net_output = self.pre_metadata_features(net_output)
        net_output = torch.cat((net_output, metadata), 1)
        net_output = self.post_metadata_features(net_output)
        net_output = self.pre_lstm_output(net_output)
        net_output = net_output.contiguous().view(batch_size, -1, 16)
        for lstm in self.lstm_encoder:
            net_output, last_hidden_cell = lstm(net_output)
            last_hidden_cell = list(last_hidden_cell)
        for lstm in self.lstm_decoder:
            if last_hidden_cell:
                # last_hidden_cell[0] = last_hidden_cell[0].contiguous().view(batch_size, -1, 256)
                # last_hidden_cell[1] = last_hidden_cell[1].contiguous().view(batch_size, -1, 256)
                net_output = lstm(self.get_decoder_seq(batch_size, self.n_steps), last_hidden_cell)[0]
                last_hidden_cell = None
            else:
                net_output = lstm(net_output)[0]
        net_output = net_output.contiguous().view(net_output.size(0), -1)
        return net_output

    def get_decoder_seq(self, batch_size, timesteps):
        decoder_input_seq = Variable(torch.zeros(batch_size, timesteps, 1))
        return decoder_input_seq.cuda() if self.is_cuda else decoder_input_seq

    def cuda(self, device_id=None):
        self.is_cuda = True
        return super(SqueezeNetTimeLSTM, self).cuda(device_id)

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])


def unit_test():
    """Tests SqueezeNetTimeLSTM for size constitency"""
    test_net = SqueezeNetTimeLSTM(6, 20)
    test_net_output = test_net(
        Variable(torch.randn(5, 36, 94, 168)),
        Variable(torch.randn(5, 12, 6, 23, 41)))
    logging.debug('Net Test Output = {}'.format(test_net_output))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())


unit_test()

Net = SqueezeNetTimeLSTM

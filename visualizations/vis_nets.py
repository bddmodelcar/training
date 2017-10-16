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


class Water(nn.Module):
    def __init__(
        self,
        contract3x3_planes,
        contract1x1_planes,
        release_planes,
        outplanes,
        outpad=(
            0,
            0)):
        super(Water, self).__init__()
        self.contract1x1_planes = contract1x1_planes
        self.outpad = outpad

        self.contract3x3_activation = nn.ReLU(inplace=True)
        self.contract3x3 = nn.ConvTranspose2d(
            contract3x3_planes, release_planes, kernel_size=3, padding=1)

        self.contract1x1_activation = nn.ReLU(inplace=True)
        self.contract1x1 = nn.ConvTranspose2d(
            contract1x1_planes, release_planes, kernel_size=1)

        self.release_activation = nn.ReLU(inplace=True)
        self.release = nn.ConvTranspose2d(
            release_planes, outplanes, kernel_size=1)

    def forward(self, x):
        x_1x1 = Variable(x.data[:, 0:self.contract1x1_planes])
        x_3x3 = Variable(x.data[:, self.contract1x1_planes:])

        x = (self.contract1x1(self.contract1x1_activation(x_1x1)) +
             self.contract3x3(self.contract3x3_activation(x_3x3))) / 2

        x = self.release(self.release_activation(x))
        if not (self.outpad == (0, 0)):
            x = torch.cat(
                (x, Variable(x.data[:, :, -1 * self.outpad[0]:, :])), 2)
            x = torch.cat(
                (x, Variable(x.data[:, :, :, -1 * self.outpad[1]:])), 3)
        return x


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        # the forward net
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
        final_conv = nn.Conv2d(512, self.N_STEPS * 2, kernel_size=1)
        self.final_output = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=6)
        )
        self.fc1 = nn.Linear(20, 1)

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

        '''
        print(x.size())
        test_func = self.final_conv
        x = test_func(x)
        print(x.size())
        '''

        x = self.final_output(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        # the reverse net
        self.N_STEPS = 10
        self.pad_final_out1 = lambda x: torch.cat(
            (x, Variable(x.data[:, :, -1:, :])), 2)
        self.pad_final_out2 = lambda x: torch.cat(
            (x, Variable(x.data[:, :, :, -3:])), 3)

        self.pre_metadata_features = nn.Sequential(
            Water(64, 64, 16, 64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 12, kernel_size=3, stride=2),
        )

        self.post_metadata_features = nn.Sequential(
            Water(256, 256, 64, 512),
            Water(256, 256, 64, 384),
            Water(192, 192, 48, 384),
            Water(192, 192, 48, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Water(128, 128, 32, 256),
            Water(128, 128, 32, 128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Water(64, 64, 26, 256, outpad=(3, 1))
        )

        beginning_conv = nn.ConvTranspose2d(
            self.N_STEPS * 2, 512, kernel_size=1)
        self.beginning_input = nn.Sequential(

            nn.UpsamplingBilinear2d(scale_factor=5),
            beginning_conv,
            nn.Dropout(p=0.5)
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if m is beginning_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], 1, 1)
        # just to set the right size for the reverse stream
        x = torch.cat((x, x), 3)
        x = self.beginning_input(x)
        x = self.post_metadata_features(x)
        x = Variable(x.data[:, 0:128])  # unconcat data
        x = self.pre_metadata_features(x)
        x = self.pad_final_out1(x)
        x = self.pad_final_out2(x)
        return x


"""AlexNet modified for regression."""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.N_FRAMES = 2
        self.N_STEPS = 10
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.post_metadata_features = nn.Sequential(
            nn.Conv2d(320, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.N_STEPS * 2),
        )

    def forward(self, x, metadata):
        metadata = metadata[:, :, 0:4, 0:9]
        x = self.pre_metadata_features(x)
        x = torch.cat((x, metadata), 1)
        x = self.post_metadata_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def unit_test():
    test_net = AlexNet()
    a = test_net(Variable(torch.randn(5, 12, 94, 168)),
                 Variable(torch.randn(5, 128, 23, 41)))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')


unit_test()

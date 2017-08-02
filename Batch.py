"""Processes data into batches for training and validation."""
from Parameters import ARGS
import numpy as np
import torch
import sys
import torch.nn.utils as nnutils
from torch.autograd import Variable

class Batch:
    def __init__(self, net):
        self.net = net
        self.outputs = None
        self.loss = None

    def forward(self, camera_data, metadata, target_data,
                optimizer, criterion):
        optimizer.zero_grad()
        self.outputs = self.net(Variable(camera_data).cuda(),
                                Variable(metadata).cuda()).cuda()
        self.loss = criterion(self.outputs, Variable(target_data.cuda()))

    def backward(self, optimizer):
        self.loss.backward()
        nnutils.clip_grad_norm(self.net.parameters(), 1.0)
        optimizer.step()

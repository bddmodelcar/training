"""Utility classes for training."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
from builtins import range
from builtins import object
import os
import operator
import time
from .Parameters import ARGS
import matplotlib.pyplot as plt
import numpy as np
import torch


class MomentCounter(object):
    """Notify after N Data Moments Passed"""

    def __init__(self, n):
        self.start = 0
        self.n = n

    def step(self, data_index):
        if data_index.ctr - self.start >= self.n:
            self.start = data_index.ctr
            return True
        return False


def csvwrite(filename, objs):
    with open(filename, 'a') as csvfile:
        csvfile.write(",".join([str(x) for x in objs]) +'\n')


class LossLog(object):
    """Keep Track of Loss, can be used within epoch or for per epoch."""

    def __init__(self):
        self.ctr = 0
        self.total_loss = 0

    def add(self, loss):
        self.total_loss += loss
        self.ctr += 1

    def average(self):
        return self.total_loss / (self.ctr * 1.)

def save_net(save_path, save_name, net):
    torch.save(
        net.state_dict(),
        os.path.join(
            save_path + save_name +
            '.weights'))

    # Next, save for inference (creates ['net'] and moves net to GPU #0)
    weights = {'net': net.state_dict().copy()}
    for key in weights['net']:
        weights['net'][key] = weights['net'][key].cuda(device=0)
    torch.save(weights,
               os.path.join(save_path + save_name + '.infer'))


def display_sort_data_moment_loss(data_moment_loss_record, data):
    sorted_data_moment_loss_record = sorted(list(data_moment_loss_record.items()),
                                            key=operator.itemgetter(1))
    low_loss_range = list(range(20))
    high_loss_range = list(range(-1, -20, -1))

    for i in low_loss_range + high_loss_range:
        l = sorted_data_moment_loss_record[i]
        run_code, seg_num, offset = sorted_data_moment_loss_record[i][0][0]
        t = sorted_data_moment_loss_record[i][0][1]
        o = sorted_data_moment_loss_record[i][0][2]

        sorted_data = data.get_data(run_code, seg_num, offset)
        plt.figure(22)
        plt.clf()
        plt.ylim(0, 1)
        plt.plot(t, 'r.')
        plt.plot(o, 'g.')
        plt.plot([0, 20], [0.5, 0.5], 'k')
        mi(sorted_data['right'][0, :, :], 23, img_title=d2s(l[1]))
        plt.pause(1)

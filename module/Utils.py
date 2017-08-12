"""Utility classes for training."""
import os
import operator
import time
from Parameters import ARGS
import matplotlib.pyplot as plt
import numpy as np
import torch


class MomentCounter:
    """Notify after N Data Moments Passed"""

    def __init__(self, n):
        self.start = 0
        self.n = n

    def step(self, ctr):
        if ctr - self.start >= self.n:
            self.start = ctr
            return True
        return False


class LossLog:
    """Keep Track of Loss, can be used within epoch or for per epoch."""

    def __init__(self):
        self.ctr = 0
        self.total_loss = 0

    def add(self, ctr, loss):
        self.total_loss += loss
        self.ctr += 1

    def average(self):
        return self.total_loss / (self.ctr * 1.)

class Timer:
    def __init__(self, time_s=0):
        self.time_s = time_s
        self.start_time = time.time()

    def check(self):
        if time.time() - self.start_time > self.time_s:
            return True
        else:
            return False

    def time(self):
        return time.time() - self.start_time

    def reset(self):
        self.start_time = time.time()

    def trigger(self):
        self.start_time = 0

class RateCounter:
    """Calculate rate of process in Hz"""

    def __init__(self):
        self.rate_ctr = 0
        self.rate_timer_interval = 10.0
        self.rate_timer = Timer(self.rate_timer_interval)

    def step(self):
        self.rate_ctr += 1
        if self.rate_timer.check():
            print('rate = ' + str(ARGS.batch_size * self.rate_ctr /
                                  self.rate_timer_interval) + 'Hz')
            self.rate_timer.reset()
            self.rate_ctr = 0

def save_net(weights_file_name, net, infer=False):
    torch.save(
        net.state_dict(),
        os.path.join(
            ARGS.save_path,
            weights_file_name +
            '.weights'))

    if infer:
    	# Next, save for inference (creates ['net'] and moves net to GPU #0)
    	weights = {'net': net.state_dict().copy()}
    	for key in weights['net']:
    	    weights['net'][key] = weights['net'][key].cuda(device=0)
    	torch.save(weights,
    	           os.path.join(ARGS.save_path, weights_file_name + '.infer'))

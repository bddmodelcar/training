"""Utility classes for training."""
import operator
import time
from Parameters import ARGS
from libs.utils2 import Timer, d2s
from libs.vis2 import mi
import matplotlib.pyplot as plt
import numpy as np
import torch


class MomentCounter:
    """Notify after N Data Moments Passed"""

    def __init__(self, n):
        self.start = 0
        self.n = n

    def step(self, data_index):
        if data_index.ctr - self.start >= self.n:
            self.start = data_index.ctr
            return True
        return False


class LossLog:
    """Keep Track of Loss, can be used within epoch or for per epoch."""

    def __init__(self):
        self.log = []
        self.ctr = 0
        self.total_loss = 0

    def add(self, ctr, loss):
        self.log.append((ctr, loss))
        self.total_loss += loss
        self.ctr += 1

    def average(self):
        return self.total_loss / (self.ctr * 1.)

    def export_csv(self, filename):
        np.savetxt(
            filename,
            np.array(self.log),
            header='Counter,Loss',
            delimiter=",",
            comments='')


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


def save_net(weights_file_name, net):
    torch.save(
        net.state_dict(),
        os.path.join(
            ARGS.save_path,
            weights_file_name +
            '.weights'))

    # Next, save for inference (creates ['net'] and moves net to GPU #0)
    weights = {'net': net.state_dict().copy()}
    for key in weights['net']:
        weights['net'][key] = weights['net'][key].cuda(device=0)
    torch.save(weights,
               ARGS.save_path + weights_file_name + '.infer')


class LossRecord:
    """ Maintain record of average loss, for intervals of 30s. """

    def __init__(self):
        self.t0 = time.time()
        self.loss_list = []
        self.timestamp_list = []
        self.loss_sum = 0
        self.loss_ctr = 0
        self.loss_timer = Timer(ARGS.loss_timer)

    def add(self, loss):
        self.loss_sum += loss
        self.loss_ctr += 1
        if self.loss_timer.check():
            self.loss_list.append(self.loss_sum / float(self.loss_ctr))
            self.loss_sum = 0
            self.loss_ctr = 0
            self.timestamp_list.append(time.time())
            self.loss_timer.reset()

    def read_csv(self, path, header=True):
        with open(path) as f:
            for line in f:
                if header:
                    header = False
                    continue
                else:
                    info = line.split(',')
                    self.timestamp_list.append(float([info[0]]))
                    self.loss_list.append([float(info[1])])

    def export_csv(self, path=None, header=True):
        csv = ''
        if header:
            csv += 'Time (s),L2 MSE Loss\n'
        for i in range(len(self.loss_list)):
            csv += str(self.timestamp_list[i] - self.t0) + ','
            csv += str(self.loss_list[i]) + '\n'

        if path is None:
            return csv
        else:
            csv_file = open(path, "wb")
            csv_file.write(csv)
            csv_file.close()

    def plot(self, color_letter='b'):
        plt.plot((np.array(self.timestamp_list) - self.t0) / 3600.0,
                 self.loss_list, color_letter + '.')


def display_sort_data_moment_loss(data_moment_loss_record, data):
    sorted_data_moment_loss_record = sorted(data_moment_loss_record.items(),
                                            key=operator.itemgetter(1))
    low_loss_range = range(20)
    high_loss_range = range(-1, -20, -1)

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

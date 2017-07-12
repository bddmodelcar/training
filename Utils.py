from Parameters import args
from libs.utils2 import *
import matplotlib.pyplot as plt
import torch


class Rate_Counter:
    """Calculate rate of process in Hz"""

    def __init__(self):
        self.rate_ctr = 0
        self.rate_timer_interval = 10.0
        self.rate_timer = Timer(self.rate_timer_interval)

    def step(self):
        self.rate_ctr += 1
        if self.rate_timer.check():
            print('rate = ' + str(args.batch_size * self.rate_ctr /
                  self.rate_timer_interval) + 'Hz')
            self.rate_timer.reset()
            self.rate_ctr = 0


def save_net(net, loss_record):
    weights_file_name = 'save_file' + time_str()
    torch.save(net.state_dict(),
               opj(args.save_path, weights_file_name + '.weights'))
    # Next, save for inference (creates ['net'] and moves net to GPU #0)
    weights = {'net': net.state_dict().copy()}
    for key in weights['net']:
        weights['net'][key] = weights['net'][key].cuda(device=0)
    torch.save(weights,
               args.save_path + weights_file_name + '.infer')


class Loss_Record:
    def __init__(self):
        self.t0 = time.time()
        self.loss_list = []
        self.timestamp_list = []
        self.loss_sum = 0
        self.loss_ctr = 0
        self.loss_timer = Timer(args.loss_timer)

    def add(self, loss):
        self.loss_sum += loss
        self.loss_ctr += 1
        if self.loss_timer.check():
            self.loss_list.append(self.loss_sum / float(self.loss_ctr))
            self.loss_sum = 0
            self.loss_ctr = 0
            self.timestamp_list.append(time.time())
            self.loss_timer.reset()

    def plot(self, color_letter='b'):
        plt.plot((np.array(self.timestamp_list) - self.t0) / 3600.0,
                 self.loss_list, color_letter + '.')


def display_sort_data_moment_loss(data_moment_loss, data):
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
        pause(1)

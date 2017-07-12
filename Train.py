from Parameters import args
import Data
import Batch
import Utils

from libs.utils2 import *
from libs.vis2 import *
import matplotlib.pyplot as plt
import operator

from nets.SqueezeNet import SqueezeNet
import torch

print(args.display)

# Set Up PyTorch Environment
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(args.gpu)
torch.cuda.device(args.gpu)

net = SqueezeNet().cuda()
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adadelta(net.parameters())

if args.resume_path is not None:
    cprint('Resuming w/ ' + args.resume_path, 'yellow')
    save_data = torch.load(args.resume_path)
    net.load_state_dict(save_data)

    loss_record_loaded = zload_obj({'path': '/home/karlzipser/loss_record'})
    loss_record = {}
    for mode in ['train', 'val']:
        loss_record[mode] = Utils.Loss_Record()
        for k in loss_record_loaded[mode].keys():
            if not callable(loss_record[mode][k]):
                loss_record[mode][k] = loss_record_loaded[mode][k]
else:
    loss_record = {}
    loss_record['train'] = Utils.Loss_Record()
    loss_record['val'] = Utils.Loss_Record()

rate_counter = Utils.Rate_Counter()

data = Data.Data()

timer = {}
timer['train'] = Timer(args.mini_train_time)
timer['val'] = Timer(args.mini_val_time)
print_timer = Timer(args.print_time)
save_timer = Timer(args.save_time)

trial_loss_record = {}

batch = Batch.Batch(net)

while True:
    for mode, data_index in [('train', data.train_index),
                             ('val', data.val_index)]:
        timer[mode].reset()
        while not timer[mode].check():

            batch.fill(data, data_index)  # Get batches ready
            batch.forward(optimizer, criterion, trial_loss_record)  # Run net, forward pass

            if mode == 'train':  # Backpropagate
                batch.backward(optimizer)

            loss_record[mode].add(batch.loss.data[0])
            rate_counter.step()

            if save_timer.check():
                Utils.save_net(net, loss_record)
                save_timer.reset()
            if print_timer.check():
                print('mode=' + mode)
                print('ctr=' + str(data_index.ctr))
                print('epoch progress=' + str(100 * data_index.ctr /
                                              len(data_index.all_steer)) + '%')
                if args.display:
                    batch.display()
                    plt.figure('loss')
                    plt.clf()  # clears figure
                    loss_record['train'].plot('b')  # plot with blue color
                    loss_record['val'].plot('r')  # plot with red color
                    print_timer.reset()

            batch = Batch.Batch(net)  # Reinitialiize batch. ***Why is this necessary?***

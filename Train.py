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

# Set Up PyTorch Environment torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(args.gpu)
torch.cuda.device(args.gpu)

net = SqueezeNet().cuda()
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adadelta(net.parameters())

if args.resume_path is not None:
    cprint('Resuming w/ ' + args.resume_path, 'yellow')
    save_data = torch.load(args.resume_path)
    net.load_state_dict(save_data)

loss_record = {}
loss_record['train'] = Utils.Loss_Record()
loss_record['val'] = Utils.Loss_Record()

rate_counter = Utils.Rate_Counter()

data = Data.Data()

# Maitains a list of all inputs to the network, and the loss and outputs for
# each of these runs. This can be used to sort the data by highest loss and
# visualize, to do so run:
# display_sort_trial_loss(data_moment_loss_record , data)
mini_train_percentage = 25
data_moment_loss_record = {}

def run_net(data_index):
    batch.fill(data, data_index)  # Get batches ready
    batch.forward(optimizer, criterion, data_moment_loss_record)

while True:
    run_net(data.train_index)  # Run network
    batch.backward(optimizer)  # Backpropagate
    train_loss_record.add(batch.loss.data[0])  # Record loss

    if ctr < 0:
        rate_counter.step()

        if data_index.epoch_complete and mode == 'train':
            
            Utils.save_net(net, loss_record, weights_prefix='epoch_' +
                           str(data_index.epoch_counter - 1) + '_')
            data_index.epoch_complete = False
            break 

        if print_timer.check():
            print(d2n('mode=',mode,
                ',ctr=',data_index.ctr,
                ',epoch progress=',dp(100*data_index.ctr / (len(data_index.valid_data_moments)*1.0)),
                ',epoch=',data_index.epoch_counter))
            if args.display:
                batch.display()
                plt.figure('loss')
                plt.clf()  # clears figure
                loss_record['train'].plot('b')  # plot with blue color
                loss_record['val'].plot('r')  # plot with red color
                print_timer.reset()

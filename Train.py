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

import traceback
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)
logging.debug(args)  # Log arguments 

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

data = Data.Data()
batch = Batch.Batch(net)

# Maitains a list of all inputs to the network, and the loss and outputs for
# each of these runs. This can be used to sort the data by highest loss and
# visualize, to do so run:
# display_sort_trial_loss(data_moment_loss_record , data)
data_moment_loss_record = {}

rate_counter = Utils.Rate_Counter()

def run_net(data_index):
    batch.fill(data, data_index)  # Get batches ready
    batch.forward(optimizer, criterion, data_moment_loss_record)

try:
    for epoch in range(1000):
        logging.debug('Starting training epoch #{}'.format(epoch))
        
        net.train()  # Train mode
        epoch_train_loss = Utils.Loss_Log()
        print_counter = Utils.Moment_Counter(500)

        while not data.train_index.epoch_complete: # Epoch of training
            run_net(data.train_index)  # Run network
            batch.backward(optimizer)  # Backpropagate

            # Logging Loss 
            epoch_train_loss.add(data.train_index.ctr, batch.loss.data[0])

            if print_counter.step(data.train_index):
                print('ctr = {}\n'
                      'most recent loss = {}\n'
                      'epoch progress = {}\n'
                      'epoch = {}\n'
                      .format(data.train_index.ctr,\
                              batch.loss.data[0],\
                              100. * data.train_index.ctr /
                              len(data.train_index.valid_data_moments),
                              epoch))

                if args.display:
                    batch.display()
                    plt.figure('loss')
                    plt.clf()  # clears figure
                    loss_record['train'].plot('b')  # plot with blue color
                    loss_record['val'].plot('r')  # plot with red color
                    print_timer.reset()

                break  # TODO: REMOVE DEBUG STATEMENT

        data.train_index.epoch_complete = False
        epoch_train_loss.export_csv('logs/epoch%02d_train_loss.csv' % (epoch,))
        logging.info('Avg Train Loss = {}'.format(epoch_train_loss.average()))
        logging.debug('Finished training epoch #{}'.format(epoch))
        logging.debug('Starting validation epoch #{}'.format(epoch))
        epoch_val_loss = Utils.Loss_Log()

        net.eval()  # Evaluate mode
        while not data.val_index.epoch_complete:
            run_net(data.train_index)  # Run network
            epoch_val_loss.add(data.train_index.ctr, batch.loss.data[0])
            break

        data.val_index.epoch_complete = False
        epoch_val_loss.export_csv('logs/epoch%02d_val_loss.csv' % (epoch,))
        logging.debug('Finished validation epoch #{}'.format(epoch))
        logging.info('Avg Val Loss = {}'.format(epoch_val_loss.average()))
        Utils.save_net("save/epoch%02d_save_%f" % (epoch,\
                       epoch_val_loss.average()), net)
except Exception:
    logging.error(traceback.format_exc())  # Log exception

    # Interrupt Saves
    Utils.save_net('save/interrupt_save.weights', net)
    epoch_train_loss.export_csv('logs/interrupt%02d_train_loss.csv' % (epoch,))
    epoch_val_loss.export_csv('logs/interrupt%02d_val_loss.csv' % (epoch,))

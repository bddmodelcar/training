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

import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)
logging.debug(args)  # Log arguments 

# Set Up PyTorch Environment torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(args.gpu)
torch.cuda.device(args.gpu)

net = SqueezeNet().cuda()
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adadelta(net.parameters())

data = Data.Data()

if args.resume_path is not None:
    cprint('Resuming w/ ' + args.resume_path, 'yellow')
    save_data = torch.load(args.resume_path)
    net.load_state_dict(save_data)

rate_counter = Utils.Rate_Counter()

def run_net(data_index):
    batch.fill(data, data_index)  # Get batches ready
    batch.forward(optimizer, criterion, data_moment_loss_record)

# TODO Try/Catch for Interrupt Save

for epoch in range(1000):
    logging.debug('Starting training epoch #{}'.format(epoch))
    
    net.train()  # Train mode
    while not data.train_index.epoch_complete:
        run_net(data.train_index)  # Run network
        batch.backward(optimizer)  # Backpropagate
        # TODO Every N moments update display
        # TODO Progress bar and log training loss
    
    logging.debug('Finished training epoch #{}'.format(epoch))
    logging.debug('Starting validation epoch #{}'.format(epoch))
    
    data.train_index.epoch_complete = False
    
    net.eval()  # Evaluate mode
    total_val_loss = val_loss_ctr = 0
    while not data.val_index.epoch_complete:
        run_net(data.train_index)  # Run network
        total_val_loss += batch.loss.data[0]
        val_loss_ctr += 1
    
    logging.debug('Finished training epoch #{}'.format(epoch))
    logging.info('Finished epoch #{}, validation loss = {}'.format(epoch,
                 total_val_loss/val_loss_ctr))

    Utils.save_net('epoch_save_...')

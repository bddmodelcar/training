"""Training and validation code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Utils
from Datasets import MergedDataset

import matplotlib.pyplot as plt

from termcolor import cprint
from nets.SqueezeNet import SqueezeNet
import torch
from torch import nn
from torch.autograd import Variable


def main():
    logging.basicConfig(filename='training.log', level=logging.DEBUG)
    logging.debug(ARGS)  # Log arguments

    # Set Up PyTorch Environment
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(ARGS.gpu)
    torch.cuda.device(ARGS.gpu)

    net = SqueezeNet().cuda()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adadelta(net.parameters())

    if ARGS.resume_path is not None:
        cprint('Resuming w/ ' + ARGS.resume_path, 'yellow')
        save_data = torch.load(ARGS.resume_path)
        net.load_state_dict(save_data)

    try:
        epoch = 0
        while True:
            logging.debug('Starting training epoch #{}'.format(epoch))

            train_dataset = MergedDataset(('/data/tpankaj/preprocess_default.hdf5',))
            train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=ARGS.batch_size,
                                                            shuffle=False, pin_memory=False,
                                                            num_workers=3)
            net.train()  # Train mode
            epoch_train_loss = Utils.LossLog()
            print_counter = Utils.MomentCounter(ARGS.print_moments)

            rate_counter = Utils.RateCounter()

            for camera_data, metadata, target_data in train_data_loader:
                camera_data = Variable(camera_data.cuda())
                metadata = Variable(metadata.cuda())
                target_data = Variable(target_data.cuda())
                # Forward Pass
                optimizer.zero_grad()
                outputs = net(camera_data, metadata).cuda()
                loss = criterion(outputs, target_data)

                # Backward Pass
                loss.backward()
                nn.utils.clip_grad_norm(net.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss.add(loss.data[0])
                rate_counter.step()

            logging.debug('Finished training epoch #{}'.format(epoch))
            Utils.csvwrite('logs/trainloss.csv',\
                           [epoch,epoch_train_loss.average()])
            del train_data_loader
            del train_dataset

            val_dataset = MergedDataset(('/data/tpankaj/preprocess_default.hdf5',),\
                                        prefix='val_')
            val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=ARGS.batch_size,
                                                          shuffle=False, pin_memory=False,
                                                          num_workers=3)
            net.eval() # Validation mode
            epoch_val_loss = Utils.LossLog()
            for camera_data, metadata, target_data in val_data_loader:
                camera_data = Variable(camera_data.cuda())
                metadata = Variable(metadata.cuda())
                target_data = Variable(target_data.cuda())
                # Forward Pass
                optimizer.zero_grad()
                outputs = net(camera_data, metadata).cuda()
                loss = criterion(outputs, target_data)

                epoch_val_loss.add(loss.data[0])
                rate_counter.step()

            Utils.save_net(
                "epoch%02d_save" %
                (epoch,), net)
            Utils.csvwrite('logs/valloss.csv',\
                           [epoch, epoch_val_loss.average()])
            del val_data_loader
            del val_dataset

            epoch += 1
    except Exception:
        logging.error(traceback.format_exc())  # Log exception
        print(traceback.format_exc())
        # Interrupt Saves
        Utils.save_net('interrupt_save', net)
if __name__ == '__main__':
    main()

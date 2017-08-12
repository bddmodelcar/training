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

    train_dataset = MergedDataset(('/hostroot/data/tpankaj/preprocess_direct.hdf5',\
                                  '/hostroot/data/tpankaj/preprocess_follow.hdf5'))

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=ARGS.batch_size,
                                              shuffle=False, pin_memory=False,
                                                    num_workers=2)


    try:
        epoch = 0
        while True:
            logging.debug('Starting training epoch #{}'.format(epoch))

            net.train()  # Train mode
            epoch_train_loss = Utils.LossLog()
            print_counter = Utils.MomentCounter(ARGS.print_moments)

            rate_counter = Utils.RateCounter()

            ctr = 0
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

                epoch_train_loss.add(ctr, loss.data[0])
                rate_counter.step()
                
                ctr += 1

            logging.debug('Finished training epoch #{}'.format(epoch))
            logging.info(
                'Avg Train Loss = {}'.format(
                    epoch_train_loss.average()))

            Utils.save_net(
                "epoch%02d_save" %
                (epoch,), net)

            epoch += 1
    except Exception:
        logging.error(traceback.format_exc())  # Log exception
        print(traceback.format_exc())
        # Interrupt Saves
        Utils.save_net('interrupt_save', net)
if __name__ == '__main__':
    main()

"""Training and validation code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Utils
from HDF5Dataset import HDF5Dataset 

import matplotlib.pyplot as plt

from termcolor import cprint
from nets.SqueezeNet import SqueezeNet
import torch


def main():
    logging.basicConfig(filename='training.log', level=logging.DEBUG)
    logging.debug(ARGS)  # Log arguments

    # Set Up PyTorch Environment
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(ARGS.gpu)
    torch.cuda.device(ARGS.gpu)

    net = SqueezeNet().cuda()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adadelta(net.parameters())

    if ARGS.resume_path is not None:
        cprint('Resuming w/ ' + ARGS.resume_path, 'yellow')
        save_data = torch.load(ARGS.resume_path)
        net.load_state_dict(save_data)

    train_dataset = MergedDataset('/data/tpankaj/preprocess_direct.hdf5',\
                                  '/data/tpankaj/preprocess_follow.hdf5')

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=ARGS.batch_size,
                                              shuffle=False, pin_memory=False,
                                              drop_last=True, num_workers=2)

    rate_counter = Utils.RateCounter()

    try:
        epoch = 0
        while True:
            logging.debug('Starting training epoch #{}'.format(epoch))

            net.train()  # Train mode
            epoch_train_loss = Utils.LossLog()
            print_counter = Utils.MomentCounter(ARGS.print_moments)

            for camera_data, metadata, target_data in train_data_loader:
                optimizer.zero_grad()
                outputs = net(camera_data, metadata).cuda()
                loss = criterion(outputs, target_data)

                epoch_train_loss.add(loss.data[0])
                rate_counter.step()

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
        # Interrupt Saves
        Utils.save_net('interrupt_save', net)
if __name__ == '__main__':
    main()

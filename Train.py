"""Training and validation code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Batch
import Utils
from HDF5Dataset import HDF5Dataset 

import matplotlib.pyplot as plt

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

    train_dataset = HDF5Dataset('/data/tpankaj/preprocess.hdf5')
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=ARGS.batch_size,
                                              shuffle=False, pin_memory=False,
                                              drop_last=True, num_workers=2)

    batch = Batch.Batch(net)
    rate_counter = Utils.RateCounter()

    try:
        epoch = -1
        avg_train_loss = Utils.LossLog()
        while True:
            logging.debug('Starting training epoch #{}'.format(epoch))

            net.train()  # Train mode
            epoch_train_loss = Utils.LossLog()
            print_counter = Utils.MomentCounter(ARGS.print_moments)
            ctr = 0

            for camera_data, metadata, target_data in train_data_loader:
                batch.forward(camera_data, metadata, target_data,
                              optimizer, criterion)
                batch.backward(optimizer)  # Backpropagate
                rate_counter.step()
                ctr += 1

            logging.info(
                'Avg Train Loss = {}'.format(
                    epoch_train_loss.average()))
            avg_train_loss.add(epoch, epoch_train_loss.average())
            avg_train_loss.export_csv('logs/avg_train_loss.csv')
            logging.debug('Finished training epoch #{}'.format(epoch))

            Utils.save_net(
                "epoch%02d_save" %
                (epoch,), net)

            epoch += 1
    except Exception:
        logging.error(traceback.format_exc())  # Log exception

        # Interrupt Saves
        Utils.save_net('interrupt_save', net)
        epoch_train_loss.export_csv(
            'logs/interrupt%02d_train_loss.csv' %
            (epoch,))

if __name__ == '__main__':
    main()

"""Training and validation code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Data
import Batch
import Utils

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

    data = Data.Data()
    batch = Batch.Batch(net)

    # Maitains a list of all inputs to the network, and the loss and outputs for
    # each of these runs. This can be used to sort the data by highest loss and
    # visualize, to do so run:
    # display_sort_trial_loss(data_moment_loss_record , data)
    data_moment_loss_record = {}
    rate_counter = Utils.RateCounter()

    def run_net(data_index):
        batch.fill(data, data_index)  # Get batches ready
        batch.forward(optimizer, criterion, data_moment_loss_record)

    try:
        epoch = 0
        avg_train_loss = Utils.LossLog()
        avg_val_loss = Utils.LossLog()
        while True:
            logging.debug('Starting training epoch #{}'.format(epoch))

            net.train()  # Train mode
            epoch_train_loss = Utils.LossLog()
            print_counter = Utils.MomentCounter(ARGS.print_moments)

            while not data.train_index.epoch_complete:  # Epoch of training
                run_net(data.train_index)  # Run network
                batch.backward(optimizer)  # Backpropagate

                # Logging Loss
                epoch_train_loss.add(data.train_index.ctr, batch.loss.data[0])
                epoch_train_loss.export_csv(
                    'logs/epoch%02d_train_loss.csv' %
                    (epoch,))

                rate_counter.step()

                if print_counter.step(data.train_index):
                    print('mode = train\n'
                          'ctr = {}\n'
                          'most recent loss = {}\n'
                          'epoch progress = {} \n'
                          'epoch = {}\n'
                          .format(data.train_index.ctr,
                                  batch.loss.data[0],
                                  100. * data.train_index.ctr /
                                  len(data.train_index.valid_data_moments),
                                  epoch))

                    if ARGS.display:
                        batch.display()
                        plt.figure('loss')
                        plt.clf()  # clears figure
                        print_timer.reset()

            data.train_index.epoch_complete = False
            logging.info(
                'Avg Train Loss = {}'.format(
                    epoch_train_loss.average()))
            avg_train_loss.add(epoch, epoch_train_loss.average())
            avg_train_loss.export_csv('logs/avg_train_loss.csv')
            logging.debug('Finished training epoch #{}'.format(epoch))
            logging.debug('Starting validation epoch #{}'.format(epoch))
            epoch_val_loss = Utils.LossLog()

            print_counter = Utils.MomentCounter(ARGS.print_moments)

            net.eval()  # Evaluate mode
            while not data.val_index.epoch_complete:
                run_net(data.val_index)  # Run network
                epoch_val_loss.add(data.train_index.ctr, batch.loss.data[0])
                epoch_val_loss.export_csv(
                    'logs/epoch%02d_val_loss.csv' %
                    (epoch,))
                print('mode = validation\n'
                      'ctr = {}\n'
                      'average val loss = {}\n'
                      'epoch progress = {} %\n'
                      'epoch = {}\n'
                      .format(data.val_index.ctr,
                              epoch_val_loss.average(),
                              100. * data.val_index.ctr /
                              len(data.val_index.valid_data_moments),
                              epoch))

            data.val_index.epoch_complete = False
            avg_val_loss.add(epoch, epoch_val_loss.average())
            avg_val_loss.export_csv('logs/avg_val_loss.csv')
            logging.debug('Finished validation epoch #{}'.format(epoch))
            logging.info('Avg Val Loss = {}'.format(epoch_val_loss.average()))
            Utils.save_net(
                "epoch%02d_save_%f" %
                (epoch, epoch_val_loss.average()), net)
            epoch += 1
    except Exception:
        logging.error(traceback.format_exc())  # Log exception

        # Interrupt Saves
        Utils.save_net('interrupt_save.weights', net)
        epoch_train_loss.export_csv(
            'logs/interrupt%02d_train_loss.csv' %
            (epoch,))
        epoch_val_loss.export_csv('logs/interrupt%02d_val_loss.csv' % (epoch,))


if __name__ == '__main__':
    main()

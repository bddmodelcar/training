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

    data = None
    batch = Batch.Batch(net)

    if ARGS.bkup is not None:
        save_data = torch.load(ARGS.bkup)
        net.load_state_dict(save_data['net'])
        data = save_data['data']
        data.get_segment_data()
        epoch = save_data['epoch']
    else:
        data = Data.Data()

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
        epoch = ARGS.epoch

        if not epoch == 0:
            import os
            print("Resuming")
            save_data = torch.load(os.path.join(ARGS.save_path, "epoch%02d.weights" % (epoch - 1,)))
            net.load_state_dict(save_data)

        logging.debug('Starting training epoch #{}'.format(epoch))

        net.train()  # Train mode
        epoch_train_loss = Utils.LossLog()
        print_counter = Utils.MomentCounter(ARGS.print_moments)
        save_counter = Utils.MomentCounter(ARGS.save_moments)

        while not data.train_index.epoch_complete:  # Epoch of training
            run_net(data.train_index)  # Run network
            batch.backward(optimizer)  # Backpropagate

            # Logging Loss
            epoch_train_loss.add(batch.loss.data[0])

            rate_counter.step()

            if save_counter.step(data.train_index):
                save_state = {'data' : data, 'net' : net.state_dict(), 'epoch' : epoch}
                if backup1:
                    torch.save(save_state, 'backup1.bkup')
                    backup1 = False
                else:
                    torch.save(save_state, 'backup2.bkup')
                    backup1 = True

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

        Utils.csvwrite('trainloss.csv', [epoch_train_loss.average()])

        logging.debug('Finished training epoch #{}'.format(epoch))
        logging.debug('Starting validation epoch #{}'.format(epoch))
        epoch_val_loss = Utils.LossLog()

        print_counter = Utils.MomentCounter(ARGS.print_moments)

        net.eval()  # Evaluate mode
        while not data.val_index.epoch_complete:
            run_net(data.val_index)  # Run network
            epoch_val_loss.add(batch.loss.data[0])

            if print_counter.step(data.val_index):
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
        Utils.csvwrite('valloss.csv', [epoch_val_loss.average()])
        logging.debug('Finished validation epoch #{}'.format(epoch))
        logging.info('Avg Val Loss = {}'.format(epoch_val_loss.average()))
        Utils.save_net("epoch%02d" % (epoch,), net)

    except Exception:
        logging.error(traceback.format_exc())  # Log exception
        # Interrupt Saves
        Utils.save_net('interrupt_save', net)


if __name__ == '__main__':
    main()

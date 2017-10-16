from Parameters import ARGS
import Data
import Batch
import Utils

from libs.utils2 import *
from libs.vis2 import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from libs import *
from nets import SqueezeNet

import traceback
import logging

import torch.nn.utils as nnutils

from vis_nets import Fire, Water, Net_D, Net_G

import scipy.misc

logging.basicConfig(filename='training.log', level=logging.DEBUG)
logging.debug(ARGS)  # Log arguments

# Set Up PyTorch Environment torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(ARGS.gpu)
torch.cuda.device(ARGS.gpu)

net = SqueezeNet.SqueezeNet().cuda()
model_path = '/home/bala/pytorch_models/epoch6goodnet'
save_data = torch.load(model_path)
net.load_state_dict(save_data['net'])
# just a dummy placeholder. doesn't really get used to update net
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adadelta(
        net.parameters())  # just a dummy placeholder. ^

net_d = Net_D().cuda()
criterion_d = torch.nn.MSELoss().cuda()
optimizer_d = torch.optim.Adadelta(net_d.parameters(), lr=.1)

net_g = Net_G().cuda()
criterion_g = torch.nn.MSELoss().cuda()
optimizer_g = torch.optim.Adadelta(net_g.parameters(), lr=10)

if ARGS.resume_path is not None:
    cprint('Resuming w/ ' + ARGS.resume_path, 'yellow')
    save_data = torch.load(ARGS.resume_path)
    net.load_state_dict(save_data)

data = Data.Data()
batch = Batch.Batch(net)
batch_d = Batch.Batch(net_d)  # this batch is based on D in GAN      

# Maitains a list of all inputs to the network, and the loss and outputs for
# each of these runs. This can be used to sort the data by highest loss and
# visualize, to do so run:
# display_sort_trial_loss(data_moment_loss_record , data)
data_moment_loss_record = {}
data_moment_loss_record_d = {}
data_moment_loss_record_g = {}

rate_counter = Utils.RateCounter()
rate_counter_d = Utils.RateCounter()
rate_counter_g = Utils.RateCounter()

def run_net(batch, optimizer, criterion, data_moment_loss_record, data_index):
    batch.fill(data, data_index)  # Get batches ready
    batch.forward(optimizer, criterion, data_moment_loss_record)

print ('start try')
intermediate_save_counter = 0

try:
    epoch = 0
    while True:
        #print('started loop')
        logging.debug('Starting training epoch #{}'.format(epoch))

        #net.train()
        net.eval() # keep pre_trained in eval mode
        net_d.train()  # Train modes for D and G networks
        net_g.train()
        
        epoch_train_loss_d = Utils.LossLog()
        epoch_train_loss_g = Utils.LossLog()
        print_counter = Utils.MomentCounter(ARGS.print_moments)

        while not data.train_index.epoch_complete:  # Epoch of training
            #print('started training d and g')
            run_net(batch, optimizer, criterion, data_moment_loss_record, data.train_index)  # Run network
            run_net(batch_d, optimizer_d, criterion_d, data_moment_loss_record_d, data.train_index)

            real_labels_d = Variable(torch.ones(ARGS.batch_size,1).cuda().float()*9.0/10)
            real_labels_g = Variable(torch.ones(ARGS.batch_size,1).cuda().float())
            fake_labels = Variable(torch.zeros(ARGS.batch_size,1).cuda().float())

            loss_d_real = criterion_d(batch_d.outputs, real_labels_d)
            metadata = torch.FloatTensor().cuda()
            zero_matrix = torch.FloatTensor(ARGS.batch_size, 1, 23, 41).zero_().cuda()
            one_matrix = torch.FloatTensor(ARGS.batch_size, 1, 23, 41).fill_(1).cuda()
            metadata = torch.cat((one_matrix, metadata), 1)
            metadata = torch.cat((one_matrix, metadata), 1)
            metadata = torch.cat((zero_matrix, metadata), 1)
            metadata = torch.cat((zero_matrix, metadata), 1)
            metadata = torch.cat((zero_matrix, metadata), 1)
            metadata = torch.cat((zero_matrix, metadata), 1)
            metadata = torch.cat(
                (torch.FloatTensor(
                    ARGS.batch_size,
                    122,
                    23,
                    41).zero_().cuda(),
                 metadata),
                1)  # Pad empty tensor
            metadata = Variable(metadata.cuda())
            fake_code = batch.outputs  # z generated from pre-trained net
            fake_images = net_g(fake_code)
            fake_outputs = net_d(fake_images, metadata)
            loss_d_fake = criterion_d(fake_outputs, fake_labels)
            
            loss_d = loss_d_real + loss_d_fake

            net_d.zero_grad()
            loss_d.backward()
            nnutils.clip_grad_norm(net_d.parameters(), 1.0)
            optimizer_d.step()

            epoch_train_loss_d.add(data.train_index.ctr, loss_d.data[0])
            rate_counter_d.step()

            ## what if I move this above loss_d.backward?? I think I should keep this here for the mini-max effect
            fake_code = batch.outputs  # z generated from pre-trained net
            fake_images = net_g(fake_code)
            # this may be slightly different from 10lines up b/c of 1 step of training
            fake_outputs = net_d(fake_images, metadata)

            loss_g = criterion_g(fake_outputs, real_labels_g)

            net_g.zero_grad()
            loss_g.backward()
            nnutils.clip_grad_norm(net_g.parameters(), 1.0)
            optimizer_g.step()

            epoch_train_loss_g.add(data.train_index.ctr, loss_g.data[0])
            rate_counter_g.step()
            
            
            #batch.backward(optimizer)  # Backpropagate

            # Logging Loss
            #epoch_train_loss.add(data.train_index.ctr, batch.loss.data[0])
            #rate_counter.step()

            if print_counter.step(data.train_index):
                print('mode = train\n'
                      'ctr = {}\n'
                      'most recent loss = {}\n'
                      'epoch progress = {} %\n'
                      'epoch = {}\n'
                      .format(data.train_index.ctr,
                              batch_d.loss.data[0],
                              100. * data.train_index.ctr /
                              len(data.train_index.valid_data_moments),
                              epoch))

                if ARGS.display:
                    batch.display()
                    plt.figure('loss')
                    plt.clf()  # clears figure
                    loss_record['train'].plot('b')  # plot with blue color
                    loss_record['val'].plot('r')  # plot with red color
                    print_timer.reset()

            if intermediate_save_counter % 5000 == 0:
                Utils.save_net(
                    "epoch%02d_save_%f_d" %
                    (epoch, epoch_train_loss_d.average()), net_d)
                Utils.save_net(
                    "epoch%02d_save_%f_g" %
                    (epoch, epoch_train_loss_g.average()), net_g)
                
            intermediate_save_counter += 1
            
        #print('epoch complete? {}'.format(data.train_index.epoch_complete))
        data.train_index.epoch_complete = False
        epoch_train_loss_d.export_csv('logs/epoch%02d_train_loss.csv' % (epoch,))
        logging.info('Avg Train Loss = {}'.format(epoch_train_loss_d.average()))
        epoch_train_loss_g.export_csv('logs/epoch%02d_train_loss.csv' % (epoch,))
        logging.info('Avg Train Loss = {}'.format(epoch_train_loss_g.average()))
        logging.debug('Finished training epoch #{}'.format(epoch))
        logging.debug('Starting validation epoch #{}'.format(epoch))
        Utils.save_net(
            "epoch%02d_save_%f_d" %
            (epoch, epoch_train_loss_d.average()), net_d)
        Utils.save_net(
            "epoch%02d_save_%f_g" %
            (epoch, epoch_train_loss_g.average()), net_g)
        
        epoch += 1



except Exception:
    logging.error(traceback.format_exc())  # Log exception
    
    # Interrupt Saves
    Utils.save_net('interrupt_save_g.weights', net_g)
    Utils.save_net('interrupt_save_d.weights', net_d)
    #epoch_train_loss_d.export_csv('logs/interrupt%02d_train_loss.csv' % (epoch,))
    #epoch_train_loss_g.export_csv('logs/interrupt%02d_train_loss.csv' % (epoch,))

'''
test_data = Variable(batch.camera_data)
test_z = net(test_data, metadata)

zero_matrix = torch.FloatTensor(args.batch_size, 1).zero_().cuda()
one_matrix = torch.FloatTensor(args.batch_size, 1).fill_(1).cuda()
test_z = torch.cat((one_matrix, zero_matrix),1)
for i in range(18):
    test_z = torch.cat((test_z, zero_matrix),1)
test_z = Variable(test_z)

test_imgs = net_g(test_z)

temp_cam_data = test_imgs.data.cpu().numpy()#[100, :, :, :]
xshape = temp_cam_data.shape[2]
yshape = temp_cam_data.shape[3]
n_frames = 2
img = np.zeros((100, xshape * n_frames, yshape * 2, 3))
for i in range(100):
    ctr = 0
    for t in range(n_frames):
        for camera in ('left', 'right'):
            for c in range(3):
                if camera == 'left':
                    img[i, t * xshape:(t + 1) * xshape, 0:yshape,
                        c] = temp_cam_data[i, ctr, :, :]
                elif camera == 'right':
                    img[i, t * xshape:(t + 1) * xshape, yshape:,
                        c] = temp_cam_data[i, ctr, :, :]
                ctr += 1
 
                
scipy.misc.imsave('test_img.png', img[0,:,:,:])
'''

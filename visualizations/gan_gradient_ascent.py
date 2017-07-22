import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np
import libs
from libs import *
import nets
from nets import SqueezeNet

from Parameters import args

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class Water(nn.Module):
    def __init__(self, contract3x3_planes, contract1x1_planes, release_planes, outplanes, outpad=(0,0)):
        super(Water, self).__init__()
        self.contract1x1_planes = contract1x1_planes
        self.outpad = outpad
        
        self.contract3x3_activation = nn.ReLU(inplace=True)
        self.contract3x3 = nn.ConvTranspose2d(contract3x3_planes, release_planes, kernel_size=3, padding=1)

        self.contract1x1_activation = nn.ReLU(inplace=True)
        self.contract1x1 = nn.ConvTranspose2d(contract1x1_planes, release_planes, kernel_size=1)

        self.release_activation = nn.ReLU(inplace=True)
        self.release = nn.ConvTranspose2d(release_planes, outplanes, kernel_size=1)

    def forward(self, x):
        x_1x1 = Variable(x.data[:,0:self.contract1x1_planes])
        x_3x3 = Variable(x.data[:,self.contract1x1_planes:])
        
        x = (self.contract1x1(self.contract1x1_activation(x_1x1)) + self.contract3x3(self.contract3x3_activation(x_3x3)))/2

        x = self.release(self.release_activation(x))
        if not (self.outpad == (0,0)):
            x = torch.cat((x,Variable(x.data[:,:,-1*self.outpad[0]:,:])),2)
            x = torch.cat((x,Variable(x.data[:,:,:,-1*self.outpad[1]:])),3)
        return x


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        ## the forward net
        self.N_STEPS = 10
        
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
        )
        self.post_metadata_features = nn.Sequential(
            Fire(256, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        final_conv = nn.Conv2d(512, self.N_STEPS * 2, kernel_size=1)
        self.final_output = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=6)
        )
        self.fc1 = nn.Linear(20,1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, metadata):
        x = self.pre_metadata_features(x)
        x = torch.cat((x, metadata), 1)
        x = self.post_metadata_features(x)

        '''
        print(x.size())
        test_func = self.final_conv
        x = test_func(x)
        print(x.size())
        '''
        
        x = self.final_output(x)
    
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        ## the reverse net
        self.N_STEPS = 10
        self.pad_final_out1 = lambda x: torch.cat((x,Variable(x.data[:,:,-1:,:])),2)
        self.pad_final_out2 = lambda x: torch.cat((x,Variable(x.data[:,:,:,-3:])),3)
        
        self.pre_metadata_features = nn.Sequential(
            Water(64, 64, 16, 64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 12, kernel_size=3, stride=2),
        )
        
        self.post_metadata_features = nn.Sequential(
            Water(256, 256, 64, 512),
            Water(256, 256, 64, 384),
            Water(192, 192, 48, 384),
            Water(192, 192, 48, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Water(128, 128, 32, 256),
            Water(128, 128, 32, 128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Water(64, 64, 26, 256, outpad=(3,1))
        )   
                

        beginning_conv = nn.ConvTranspose2d(self.N_STEPS * 2, 512, kernel_size=1)
        self.beginning_input = nn.Sequential(
            
            nn.UpsamplingBilinear2d(scale_factor=5),
            beginning_conv,
            nn.Dropout(p=0.5)
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if m is beginning_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()                                                                                                

    def forward(self, x):
        x = x.view(x.size()[0],x.size()[1],1,1)
        x = torch.cat((x,x),3) # just to set the right size for the reverse stream
        x = self.beginning_input(x)
        x = self.post_metadata_features(x)
        x = Variable(x.data[:,0:128]) #unconcat data
        x = self.pre_metadata_features(x)
        x = self.pad_final_out1(x)
        x = self.pad_final_out2(x)
        return x


# Set Up PyTorch Environment torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(args.gpu)
torch.cuda.device(args.gpu)

net = SqueezeNet.SqueezeNet().cuda()
model_path = '/home/bala/pytorch_models/epoch6goodnet'
save_data = torch.load(model_path)
net.load_state_dict(save_data['net'])
criterion = torch.nn.MSELoss().cuda() ## just a dummy placeholder. doesn't really get used to update net
optimizer = torch.optim.Adadelta(net.parameters()) ## just a dummy placeholder. ^

net_d = Net_D().cuda()
criterion_d = torch.nn.MSELoss().cuda()
optimizer_d = torch.optim.Adadelta(net.parameters())

net_g = Net_G().cuda()
criterion_g = torch.nn.MSELoss().cuda()
optimizer_g = torch.optim.Adadelta(net.parameters())

if args.resume_path is not None:
    cprint('Resuming w/ ' + args.resume_path, 'yellow')
    save_data = torch.load(args.resume_path)
    net.load_state_dict(save_data)

loss_record_d = {}
loss_record_d['train'] = Utils.Loss_Record()
loss_record_d['val'] = Utils.Loss_Record()

loss_record_g = {}
loss_record_g['train'] = Utils.Loss_Record()
loss_record_g['val'] = Utils.Loss_Record()

rate_counter_d = Utils.Rate_Counter()

rate_counter_g = Utils.Rate_Counter() 

data = Data.Data()

timer = {}
timer['train'] = Timer(args.mini_train_time)
timer['val'] = Timer(args.mini_val_time)
print_timer = Timer(args.print_time)
save_timer = Timer(args.save_time)

# Maintains a list of all inputs to the network, and the loss and outputs for
# each of these runs.
data_moment_loss_record = {}
data_moment_loss_record_d = {}
data_moment_loss_record_g = {}

batch = Batch.Batch(net) ## this batch is based on the pretrained net, used for generating realistic z's for GAN
batch_d = Batch.Batch(net_d) ## this batch is based on D in GAN 

while True:
    for mode, data_index in [('train', data.train_index),
                             ('val', data.val_index)]:
        timer[mode].reset()
        while not timer[mode].check():

            batch.fill(data,data_index)
            batch_d.fill(data, data_index)  # Get batches ready
            
            real_labels = to_var(torch.ones(args.batch_size))
            fake_labels = to_var(torch.zeros(args.batch_size))

            # Run net, forward pass
            batch.forward(optimizer, criterion, data_moment_loss_record)
            batch_d.forward(optimizer_d, criterion_d, data_moment_loss_record_d)
            loss_d_real = criterion(batch_d.outputs, real_labels)

            fake_code = batch.outputs ## z generated from pre-trained net
            fake_images = net_g(fake_code)
            fake_outputs = net_d(fake_images)
            loss_d_fake = criterion_d(fake_outputs, fake_labels)
            
            loss_d = loss_d_real + loss_d_fake

            if mode == 'train':
                net_d.zero_grad()
                loss_d.backward()
                nnutils.clip_grad_norm(net_d.parameters(), 1.0)
                optimizer_d.step()
            
            #if mode == 'train':  # Backpropagate
            #    batch.backward(optimizer)

            loss_record_d[mode].add(loss_d.data[0])
            rate_counter_d.step()

            ### what if I move this above loss_d.backward?? I think I should keep this here for the mini-max effect
            fake_code = batch.outputs ## z generated from pre-trained net
            fake_images = net_g(fake_code)
            fake_outputs = net_d(fake_images) ## this may be slightly different from 10lines up b/c of 1 step of training

            loss_g = criterion_g(fake_outputs, real_labels)

            if mode == 'train':
                net_g.zero_grad()
                loss_g.backward()
                nnutils.clip_grad_norm(net_g.parameters(), 1.0)
                optimizer_g.step()

            loss_record_g[mode].add(loss_g.data[0])
            rate_counter_g.step()
            
            if save_timer.check():
                Utils.save_net(net_d, loss_record_d)
                Utils.save_net(net_g, loss_record_g)
                save_timer.reset()

            if mode == 'train' and data_index.epoch_complete:
                Utils.save_net(net_d, loss_record, weights_prefix='epoch_' +
                               str(data_index.epoch_counter - 1) + '_')
                Utils.save_net(net_g, loss_record, weights_prefix='epoch_' +
                               str(data_index.epoch_counter - 1) + '_')
                data_index.epoch_complete = False

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
                    

    
'''
x = Variable(torch.randn(1,12,94,168))
metadata = Variable(torch.randn(1,128,23,41))
x = net_d(x,metadata)
x = net_g(x)

temp_cam_data = x.data.numpy()[0,:,:,:]
xshape = temp_cam_data.shape[1]; yshape = temp_cam_data.shape[2]
n_frames = 2
img = np.zeros((xshape*n_frames,yshape*2,3))
ctr = 0

for t in range(n_frames):
    for camera in ('left','right'):
        for c in range(3):
            if camera == 'left':
                img[t*xshape:(t+1)*xshape, 0:yshape, c] = temp_cam_data[ctr,:,:]
            elif camera == 'right':
                img[t*xshape:(t+1)*xshape, yshape:, c] = temp_cam_data[ctr,:,:]
            ctr += 1

#scipy.misc.imsave('rf_imgs/gan1.png', img)
'''

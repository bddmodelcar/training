import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as nnutils
from torch.autograd import Variable
from libs.import_utils import *
import nets
from nets.squeezenet import SqueezeNet

net = SqueezeNet().cuda()
criterion = nn.MSELoss().cuda()  # define loss function
optimizer = torch.optim.Adadelta(net.parameters())

model_path = '/home/schowdhuri/working-directory/training_grounds/newinput-squeezenet/oldsave/directfollow6epochs/epoch_save_7.0.00249056216403'
save_data = torch.load(model_path)
net.load_state_dict(save_data['net'])

#########
''' manually arange the network functions in a list'''
#########
all_layer_funcs = []
for i in range(4): # custom size because i know the shape of this sequential
    all_layer_funcs.append(net.pre_metadata_features[i])
    
all_layer_funcs.append(lambda x:torch.cat((x, meta_data), 1))

for i in range(9): # custom size because i know the shape of this sequential
    all_layer_funcs.append(net.post_metadata_features[i])

all_layer_funcs.append(net.final_output)

all_layer_funcs.append(lambda x:x.view(x.size(0), -1))

layer_to_vis = len(all_layer_funcs)-1 ## SHOULD BE A CONTROLLABLE PARAMETER
camera_data = Variable(torch.randn(1,12,94,168).cuda(), requires_grad=True)
meta_data = Variable(torch.randn(1,128,23,41).cuda(), requires_grad=True)
output = camera_data
for i,layer in enumerate(all_layer_funcs):
    output = layer(output)
    if i == layer_to_vis:
        break
    
dims_to_vis = np.arange(output.size()[1])
for dim_to_vis in dims_to_vis:
    camera_data = Variable(torch.randn(1,12,94,168).cuda(), requires_grad=True) #RESETS FOR EACH DIM
    meta_data = Variable(torch.randn(1,128,23,41).cuda(), requires_grad=True)
    output = camera_data # NEED TO RECALC SINCE INPUT DATA RESET
    for i,layer in enumerate(all_layer_funcs):
        output = layer(output)
        if i == layer_to_vis:
            break

    batch_size = output.size()[0]; dim_size = output.size()[1]
    print(output.size())
    if len(output.size())>2:
        height = output.size()[2]; width = output.size()[3]
        grad_temp0 = torch.zeros(batch_size, dim_to_vis, height, width).cuda()
        grad_temp1 = torch.ones(batch_size, 1, height, width).cuda() #this is the dim that gets visualized, all others' gradients set to 0
        grad_temp2 = torch.zeros(batch_size, dim_size-dim_to_vis-1, height, width).cuda()
    else:
        grad_temp0 = torch.ones(batch_size, dim_to_vis).cuda()
        grad_temp1 = torch.ones(batch_size, 1).cuda()
        grad_temp2 = torch.zeros(batch_size, dim_size-dim_to_vis-1).cuda()

    if dim_to_vis == 0:
        gradients = torch.cat((grad_temp1, grad_temp2),dim=1)
    elif dim_to_vis == output.size()[1]-1:
        gradients = torch.cat((grad_temp0, grad_temp1),dim=1)
    else:
        gradients = torch.cat((grad_temp0, grad_temp1, grad_temp2),dim=1)
    
    print('dim {}\'s image gradient: {}'.format(dim_to_vis,camera_data.grad))
    output.backward(gradients)
    print('dim {}\'s image gradient: {}'.format(dim_to_vis,camera_data.grad))
    
'''
output = all_layer_funcs[0](camera_data)
grad_temp1 = torch.ones(1,1,46,83).cuda()
grad_temp2 = torch.zeros(1,63,46,83).cuda()
gradients = torch.cat((grad_temp1, grad_temp2),dim=1)
print(output.size())
output.backward(gradients)
print(camera_data.grad)
'''

'''
grad_temp1 = torch.ones(1,1).cuda()
grad_temp2 = torch.zeros(1,19).cuda()
gradients = torch.cat((grad_temp1, grad_temp2),dim=1) 
'''

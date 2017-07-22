import numpy as np

import torch
from torch.autograd import Variable
from libs import *
from nets import SqueezeNet

import scipy.misc

print('about get squeezenet architecture')
net = SqueezeNet.SqueezeNet().cuda()
# criterion = nn.MSELoss().cuda()  # define loss function
#optimizer = torch.optim.Adadelta(net.parameters())
model_path = '/home/bala/pytorch_models/epoch6goodnet'
print('about to load in squeezenet model')
save_data = torch.load(model_path)
net.load_state_dict(save_data['net'])

#########
""" manually arange the network functions in a list """
#########
print('about to set up functions')
all_layer_funcs = []
for i in range(4):  # custom size because i know the shape of this sequential
    all_layer_funcs.append(net.pre_metadata_features[i])

all_layer_funcs.append(lambda x: torch.cat((x, metadata), 1))

for i in range(9):  # custom size because i know the shape of this sequential
    all_layer_funcs.append(net.post_metadata_features[i])

all_layer_funcs.append(net.final_output)

all_layer_funcs.append(lambda x: x.view(x.size(0), -1))

# len(all_layer_funcs)-1 ## SHOULD BE A CONTROLLABLE PARAMETER
layer_to_vis = 6

camera_data = Variable(torch.randn(1, 12, 94, 168).cuda(), requires_grad=True)
metadata = Variable(torch.randn(1, 128, 23, 41).cuda(), requires_grad=True)
output = camera_data
for i, layer in enumerate(all_layer_funcs):
    output = layer(output)
    if i == layer_to_vis:
        break

print('about to start visualizing data')

dims_to_vis = np.arange(output.size()[1])  # channel
learning_rate = 1
for dim_to_vis in dims_to_vis:
    camera_data = Variable(
        255 *
        torch.randn(
            1,
            12,
            94,
            168).cuda(),
        requires_grad=True)  # RESETS FOR EACH DIM SO NOT VOLATILE

    metadata = torch.FloatTensor().cuda()

    zero_matrix = torch.FloatTensor(1, 1, 23, 41).zero_().cuda()
    one_matrix = torch.FloatTensor(1, 1, 23, 41).fill_(1).cuda()

    metadata = torch.cat((one_matrix, metadata), 1)
    metadata = torch.cat((one_matrix, metadata), 1)
    metadata = torch.cat((zero_matrix, metadata), 1)
    metadata = torch.cat((zero_matrix, metadata), 1)
    metadata = torch.cat((zero_matrix, metadata), 1)
    metadata = torch.cat((zero_matrix, metadata), 1)
    metadata = torch.cat(
        (torch.FloatTensor(
            1,
            122,
            23,
            41).zero_().cuda(),
            metadata),
        1)  # Pad empty tensor
    metadata = Variable(metadata.cuda())

    for asc_iter in range(1000):
        jitter = 32
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        jittered_camera_data = np.roll(
            np.roll(camera_data.data.cpu().numpy(), ox, -1), oy, -2)
        # performing jitter is convenient b/c it also resets Variable so it's
        # not volatile for back-prop
        camera_data = Variable(
            torch.from_numpy(jittered_camera_data).cuda(),
            requires_grad=True)

        output = camera_data  # NEED TO RECALC SINCE INPUT DATA RESET
        for i, layer in enumerate(all_layer_funcs):
            output = layer(output)
            if i == layer_to_vis:
                break

        batch_size = output.size()[0]
        dim_size = output.size()[1]

        if len(output.size()) > 2:
            height = output.size()[2]
            width = output.size()[3]
            grad_temp0 = torch.zeros(
                batch_size, dim_to_vis, height, width).cuda()
            # this is the dim that gets visualized, all others' gradients set
            # to 0
            grad_temp1 = torch.ones(batch_size, 1, height, width).cuda()
            grad_temp2 = torch.zeros(
                batch_size,
                dim_size - dim_to_vis - 1,
                height,
                width).cuda()
        else:
            grad_temp0 = torch.ones(batch_size, dim_to_vis).cuda()
            grad_temp1 = torch.ones(batch_size, 1).cuda()
            grad_temp2 = torch.zeros(
                batch_size, dim_size - dim_to_vis - 1).cuda()

        if dim_to_vis == 0:
            gradients = torch.cat((grad_temp1, grad_temp2), dim=1)
        elif dim_to_vis == output.size()[1] - 1:
            gradients = torch.cat((grad_temp0, grad_temp1), dim=1)
        else:
            gradients = torch.cat((grad_temp0, grad_temp1, grad_temp2), dim=1)

        output.backward(gradients)

        g = camera_data.grad.data.cpu().numpy()
        g_abs_mean = np.abs(g.flatten()).mean()
        camera_data.data.add_(
            torch.from_numpy(
                g *
                learning_rate /
                g_abs_mean).cuda())

        if (asc_iter + 1) % 100 == 0:
            print('layer_{}_channel_{}_iter_{}.png'.format(
                layer_to_vis, dim_to_vis, asc_iter + 1))

            temp_cam_data = camera_data.data.cpu().numpy()[0, :, :, :]
            xshape = temp_cam_data.shape[1]
            yshape = temp_cam_data.shape[2]
            n_frames = 2
            img = np.zeros((xshape * n_frames, yshape * 2, 3))
            ctr = 0

            for t in range(n_frames):
                for camera in ('left', 'right'):
                    for c in range(3):
                        if camera == 'left':
                            img[t * xshape:(t + 1) * xshape, 0:yshape,
                                c] = temp_cam_data[ctr, :, :]
                        elif camera == 'right':
                            img[t * xshape:(t + 1) * xshape, yshape:,
                                c] = temp_cam_data[ctr, :, :]
                        ctr += 1

            scipy.misc.imsave(
                'rf_imgs/layer_{}_channel_{}_iter_{}.png'.format(
                    layer_to_vis, dim_to_vis, asc_iter + 1), img)

        unjittered_camera_data = np.roll(
            np.roll(camera_data.data.cpu().numpy(), -1 * ox, -1), -1 * oy, -2)
        camera_data = Variable(
            torch.from_numpy(unjittered_camera_data).cuda(),
            requires_grad=True)


"""
output = all_layer_funcs[0](camera_data)
grad_temp1 = torch.ones(1,1,46,83).cuda()
grad_temp2 = torch.zeros(1,63,46,83).cuda()
gradients = torch.cat((grad_temp1, grad_temp2),dim=1)
print(output.size())
output.backward(gradients)
print(camera_data.grad)
"""

"""
grad_temp1 = torch.ones(1,1).cuda()
grad_temp2 = torch.zeros(1,19).cuda()
gradients = torch.cat((grad_temp1, grad_temp2),dim=1)
"""

"""Processes data into batches for preprocessing."""
from Parameters import ARGS
from libs.utils2 import z2o
from libs.vis2 import mi
import numpy as np
import torch
import sys
import torch.nn.utils as nnutils
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Batch:

    def clear(self):
        ''' Clears batch variables before forward pass '''
        self.camera_data = torch.FloatTensor().cuda()
        self.metadata = torch.FloatTensor().cuda()
        self.target_data = torch.FloatTensor().cuda()
        self.names = []
        self.outputs = None
        self.loss = None

    def __init__(self):
        self.camera_data = None
        self.metadata = None
        self.target_data = None
        self.names = None
        self.outputs = None
        self.loss = None
        self.data_ids = None

    def fill(self, data, data_index):
        self.clear()
        self.data_ids = []
        self.camera_data = torch.ByteTensor(
            ARGS.batch_size, ARGS.nframes * 6, 94, 168).cuda()
        self.metadata = torch.ByteTensor(ARGS.batch_size, 128, 23, 41).cuda()
        self.target_data = torch.ByteTensor(ARGS.batch_size, 20).cuda()
        for data_number in range(ARGS.batch_size):
            data_point = None
            while data_point is None:
                e = data.next(data_index)
                run_code = e[3]
                seg_num = e[0]
                offset = e[1]
                data_point = data.get_data(run_code, seg_num, offset)

            self.data_ids.append((run_code, seg_num, offset))
            self.data_into_batch(data_point, data_number)
        return (self.camera_data, self.metadata, self.target_data)

    def data_into_batch(self, data, data_number):
        self.names.insert(0, data['name'])

        # Convert Camera Data to PyTorch Ready Tensors
        list_camera_input = []
        for t in range(ARGS.nframes):
            for camera in ('left', 'right'):
                list_camera_input.append(torch.from_numpy(data[camera][t]))
        camera_data = torch.cat(list_camera_input, 2)
        camera_data = camera_data.cuda()#.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)
        self.camera_data[data_number, :, :, :] = camera_data

        # Convert Behavioral Modes/Metadata to PyTorch Ready Tensors
        metadata = torch.ByteTensor(128, 23, 41).cuda()
        zero_matrix = torch.ByteTensor(23, 41).zero_().cuda()
        one_matrix = torch.ByteTensor(23, 41).fill_(1).cuda()
        metadata_count = 127
        for cur_label in ['racing', 'caffe', 'follow', 'direct', 'play',
                          'furtive']:
            if cur_label == 'caffe':
                if data['states'][0]:
                    metadata[metadata_count, :, :] = one_matrix
                else:
                    metadata[metadata_count, :, :] = zero_matrix
            else:
                if data['labels'][cur_label]:
                    metadata[metadata_count, :, :] = one_matrix
                else:
                    metadata[metadata_count, :, :] = zero_matrix
            metadata_count -= 1
        metadata[0:122, :, :] = torch.ByteTensor(
            122, 23, 41).zero_().cuda()  # Pad empty tensor
        self.metadata[data_number, :, :, :] = metadata

        # Figure out which timesteps of labels to get
        s = data['steer']
        m = data['motor']
        r = range(ARGS.stride * ARGS.nsteps - 1, -1, -ARGS.stride)[::-1]
        s = np.array(s)[r]
        m = np.array(m)[r]

        # Convert labels to PyTorch Ready Tensors
        steer = torch.from_numpy(s).cuda()#.float() / 99.
        motor = torch.from_numpy(m).cuda()#.float() / 99.
        target_data = torch.ByteTensor(steer.size()[0] + motor.size()[0])
        target_data[0:steer.size()[0]] = steer
        target_data[steer.size()[0]:steer.size()[0] + motor.size()[0]] = motor
        self.target_data[data_number, :] = target_data

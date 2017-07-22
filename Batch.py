"""Processes data into batches for training and validation."""
from Parameters import args
from lib.utils2 import z2o
from lib.utils2 import mi
import numpy as np
import torch
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

    def __init__(self, net):
        self.net = net
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
        for _ in range(args.batch_size):
            data_point = None
            while data_point is None:
                e = data.next(data_index)
                run_code = e[3]
                seg_num = e[0]
                offset = e[1]
                data_point = data.get_data(run_code, seg_num, offset)

            self.data_ids.append((run_code, seg_num, offset))
            self.data_into_batch(data_point)

    def data_into_batch(self, data):
        self.names.insert(0, data['name'])

        # Convert Camera Data to PyTorch Ready Tensors
        list_camera_input = []
        for t in range(args.nframes):
            for camera in ('left', 'right'):
                list_camera_input.append(torch.from_numpy(data[camera][t]))
        camera_data = torch.cat(list_camera_input, 2)
        camera_data = camera_data.cuda().float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)
        self.camera_data = torch.cat((self.camera_data,
                                      torch.unsqueeze(camera_data, 0)), 0)

        # Convert Behavioral Modes/Metadata to PyTorch Ready Tensors
        metadata = torch.FloatTensor().cuda()
        zero_matrix = torch.FloatTensor(1, 1, 23, 41).zero_().cuda()
        one_matrix = torch.FloatTensor(1, 1, 23, 41).fill_(1).cuda()
        for cur_label in ['racing', 'caffe', 'follow', 'direct', 'play',
                          'furtive']:
            if cur_label == 'caffe':
                if data['states'][0]:
                    metadata = torch.cat((one_matrix, metadata), 1)
                else:
                    metadata = torch.cat((zero_matrix, metadata), 1)
            else:
                if data['labels'][cur_label]:
                    metadata = torch.cat((one_matrix, metadata), 1)
                else:
                    metadata = torch.cat((zero_matrix, metadata), 1)
        metadata = torch.cat((torch.FloatTensor(1, 122, 23, 41).zero_().cuda(),
                              metadata), 1)  # Pad empty tensor
        self.metadata = torch.cat((self.metadata, metadata), 0)

        # Figure out which timesteps of labels to get
        s = data['steer']
        m = data['motor']
        r = range(args.stride * args.nsteps - 1, -1, -args.stride)[::-1]
        s = np.array(s)[r]
        m = np.array(m)[r]

        # Convert labels to PyTorch Ready Tensors
        steer = torch.from_numpy(s).cuda().float() / 99.
        motor = torch.from_numpy(m).cuda().float() / 99.
        target_data = torch.unsqueeze(torch.cat((steer, motor), 0), 0)
        self.target_data = torch.cat((self.target_data, target_data), 0)

    def forward(self, optimizer, criterion, data_moment_loss_record):
        optimizer.zero_grad()
        self.outputs = self.net(Variable(self.camera_data),
                                Variable(self.metadata)).cuda()
        self.loss = criterion(self.outputs, Variable(self.target_data))

        for b in range(args.batch_size):
            data_id = self.data_ids[b]
            t = self.target_data[b].cpu().numpy()
            o = self.outputs[b].data.cpu().numpy()
            a = (self.target_data[b] - self.outputs[b].data).cpu().numpy()
            loss = np.sqrt(a * a).mean()
            data_moment_loss_record[(data_id, tuple(t), tuple(o))] = loss

    def backward(self, optimizer):
        self.loss.backward()
        nnutils.clip_grad_norm(self.net.parameters(), 1.0)
        optimizer.step()

    def display(self):
        if args.display:
            o = self.outputs[0].data.cpu().numpy()
            t = self.target_data[0].cpu().numpy()

            print('Loss:', np.round(self.loss.data.cpu().numpy()[0], decimals=5))
            a = self.camera_data[0][:].cpu().numpy()
            b = a.transpose(1, 2, 0)
            h = np.shape(a)[1]
            w = np.shape(a)[2]
            c = np.zeros((10 + h * 2, 10 + 2 * w, 3))
            c[:h, :w, :] = z2o(b[:, :, 3:6])
            c[:h, -w:, :] = z2o(b[:, :, :3])
            c[-h:, :w, :] = z2o(b[:, :, 9:12])
            c[-h:, -w:, :] = z2o(b[:, :, 6:9])
            mi(c, 'cameras')
            print(a.min(), a.max())
            plt.figure('steer')
            plt.clf()
            plt.ylim(-0.05, 1.05)
            plt.xlim(0, len(t))
            plt.plot([-1, 60], [0.49, 0.49], 'k')
            plt.plot(o, 'og')
            plt.plot(t, 'or')
            plt.title(self.names[0])
            plt.pause(0.000000001)

import numpy as np
import h5py
import torch
import os



class Dataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, n_frames):
        self.dir = data_dir
        self.n_frames = n_frames
        self.moments = []
        self.left_imgs = []
        self.right_imgs = []
        self.steer_cmds = []
        self.throttle_cmds = []
        self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for filename in self.sort_filelist(self.data_dir):
            hdf5_file = h5py.File(filename, 'r')
            left_data = hdf5_file.get('left')
            right_data = hdf5_file.get('right')
            steer_data = hdf5_file.get('steer')
            throttle_data = hdf5_file.get('throttle')

            for i in range(len(left_data)):
                self.left_imgs.append(left_data[i])
                self.right_imgs.append(right_data[i])
                self.steer_cmds.append(steer_data[i])
                self.throttle_cmds.append(throttle_data[i])

                # every n_frames, create data moment
                if (i + 1) % n_frames == 0:
                    moments.append(DataMoment(self.left_imgs, 
                                            self.right_imgs, 
                                            self.steer_cmds, 
                                            self.throttle_cmds, 
                                            None, 
                                            i, 
                                            self.num_frames))
                    self.left_imgs, self.right_imgs, self.steer_cmds, self.throttle_cmds = [], [], [], []



	def sort_filelist(self, data_dir):
        file_list = []
        for file in os.listdir(data_dir):
            if file_name.endswith('hdf5'):
                file_list.append(file_name)
        return sorted(file_list) # TODO: check in what order files get sorted


    def __len__(self):
        return len(self.moments)

    def __getitem__(self, index):

        moment = self.moments[index]
        camera_data = []
        print('camera_data should be cuda. Currently it is', camera_data.device)

        for i in range(self.n_frames):
            camera_data.append(torch.from_numpy(moment.left_imgs[i]).to(torch.device)) # is it okay to use numpy instead of float?
            camera_data.append(torch.from_numpy(moment.right_imgs[i]).to(torch.device))
            print('img is what kind of tensor?:', type(camera_data[-1]), camera_data[-1].device)
            
        camera_data = torch.cat((camera_data, img), 2)
        print('camera_data is what kind of tensor?:', type(camera_data, camera_data.device))
        camera_data = camera_data.cuda().float() / 255. - 0.5 # maybe remove cuda?
        camera_data = torch.transpose(camera_data, 0, 2) 
        camera_data = torch.transpose(camera_data, 1, 2)
        # why is the unsqueeze and scaling not occuring here, like it does in dataFormatter.py?

        all_steers = torch.Tensor(moment.steer_cmds).to(self.dev)
        all_throttles = torch.Tensor(moment.throttle_cmds).to(self.dev)

        return camera_data, all_steers, all_throttles


class DataMoment():
    def __init__(self, left_imgs, right_imgs, steers, throttles, start_index=None, end_index, self.num_frames):
        self.left_imgs = left_imgs
        self.right_imgs = right_imgs
        self.steers = steers
        self.throttles = throttles
        self.end_index = end_index
        self.num_frames
        if not start_index: # do I need start and end index??
            self.start_index = end_index - num_frames
        else:
            self.start_index = start_index

        # TODO: stack images in a moment, right?


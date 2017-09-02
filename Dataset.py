import numpy as np
import h5py
import torch
import torch.utils.data as data
import sys
import h5py
from scipy import signal
from random import shuffle
import os

data_dirs = os.walk('/hostroot/home/dataset/data_2017_08_29/bdd_aruco_demo/'
                    'h5py').next()[1]

class Dataset(data.Dataset):
    def __init__(self, data_folder_dir, run_skip = []):
        self.runs = os.walk(os.path.join(data_folder_dir, 'h5py')).next()[1]
        self.run_files = []
        
        # Initialize List of Files
        self.shuffle_runs()
	self.run_list = [0]
        self.total_length = 0
	for run in self.runs:
            if run in run_skip:
                continue
	    f = h5py.File(os.path.join(data_folder_dir, 'h5py', run), 'r')
            length = f['left_image_flip']['vals'].shape[0]

            self.run_files.append(f)
            self.run_list.append(total_length)
            self.total_length += length

        self.run_list = self.run_list[:-1] # Get rid of last element (speed)


    def __getitem__(self, index)
        run_idx, time_idx = self.create_map(index)

        #TODO: Deal with nsteps nframes and stride somehow
        # Can prob reuse frames in diff data moments with different
        # starting time_idx
        img = run_list[run_idx]['left_image_flip'][time_idx, :, :, :] 

    def create_map(self, global_index):
        for idx, length in enumerate(self.run_list[::-1]):
            if global_index >= length:
                return idx, global_index - length

    def shuffle_runs(self):
        shuffle(self.runs)

# show an image
img = f['left_image_flip']['vals'][0, :, :, :]

import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('image', cv2.cv.CV_WINDOW_NORMAL)

# create trackbars for color change
# img = f['left_image_flip']['vals'][timestep, :, :, :]

cv2.destroyAllWindows()

class MergedDataset(data.Dataset):
 
    def __init__(self, hdf5_list, prefix='train_', equalize=False):
        self.prefix = prefix
        self.equalize = equalize
        self.datasets = []
        self.start = []
        self.end = []
        self.total_count = 0
        self.minlen = float("inf")
        for f in hdf5_list:
            print(f)
            h5_file = h5py.File(f, 'r')
            self.datasets.append(h5_file)
            self.start.append(self.total_count)
            data_len = h5_file[prefix+'camera_data'].shape[0]
            self.total_count += data_len
            self.minlen = min(data_len, self.minlen)
            self.end.append(self.total_count)

        if equalize:
            self.total_count = len(hdf5_list) * self.minlen


    def __getitem__(self, index):
        for idx, lim in enumerate(self.end):
            if (self.equalize and (idx + 1) * self.minlen > index)\
                    or lim > index:
                datanum = idx
                break
        dataset = self.datasets[datanum]
        index -= self.start[datanum]
        camera_data = dataset[self.prefix+'camera_data'][index, :, :, :]
        metadata = dataset[self.prefix+'metadata'][index, :, :, :]
        target_data = dataset[self.prefix+'target_data'][index, :]
        camera_data = torch.from_numpy(camera_data.astype('float32') / 255. - 0.5)
        metadata = torch.from_numpy(metadata.astype('float32'))
        target_data = torch.from_numpy(target_data.astype('float32') / 99.)
        return camera_data, metadata, target_data

    def __len__(self):
        return self.total_count

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
    def __init__(self, data_folder_dir, run_skip = [], stride=3):
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
            self.run_list.append(total_length - 7)  # Get rid of the first 7 frames as starting points
            self.total_length += (length - 106)

	self.run_list = self.run_list[:-1] # Get rid of last element (speed)

	# Create row gradient
        self.row_gradient = torch.FloatTensor(94, 168)
        for row in range(94):
            self.row_gradient[row, :] = row / 93.

        # Create col gradient
        self.col_gradient = torch.FloatTensor(94, 168)
        for col in range(168):
            self.col_gradient[:, col] = col / 167.

        self.time_map = range(0, stride * 10, stride)


    def __getitem__(self, index)
        run_idx, time_idx = self.create_map(index)

        #TODO: Deal with nsteps nframes and stride somehow
        # Can prob reuse frames in diff data moments with different
        # starting time_idx and use an array to translate t into the idx

        # Convert Camera Data to PyTorch Ready Tensors
        img = run_files[run_idx]['left_image_flip'][t, :, :, :]
        img = run_files[run_idx]['left_image_flip'][t, :, :, :]

        list_camera_input = []
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-7]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-6,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-5,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-4,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-3,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-2,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t-1,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['left_image_flip'][t,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['right_image_flip'][t-1,:,:,1:2]))
        list_camera_input.append(torch.from_numpy(run_files[run_idx]['right_image_flip'][t,:,:,1:2]))

        camera_data = torch.cat(list_camera_input, 2)
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)

        final_camera_data = torch.FloatTensor()
        final_camera_data[data_number, 0:12, :, :] = camera_data
        final_camera_data[data_number, 12, :, :] = self.row_gradient
        final_camera_data[data_number, 13, :, :] = self.col_gradient

        return final_camera_data
	

    def create_map(self, global_index):
        for idx, length in enumerate(self.run_list[::-1]):
            if global_index >= length:
                return idx, global_index - length

    def shuffle_runs(self):
        shuffle(self.runs)

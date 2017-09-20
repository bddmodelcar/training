import numpy as np
import time
import h5py
import torch
import torch.utils.data as data
import sys
from random import shuffle
import os
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, data_folder_dir, require_one, ignore_list, stride=10, max_len=-1):
        self.max_len = max_len
	self.runs = os.walk(os.path.join(data_folder_dir, 'processed_h5py'), followlinks=True).next()[1]
        shuffle(self.runs)  # shuffle each epoch to allow shuffle False
        self.run_files = []

        # Initialize List of Files
        self.invisible = []
        self.visible = []
        self.total_length = 0 
        self.full_length = 0 


        for run in self.runs:
            segs_in_run = os.walk(os.path.join(data_folder_dir, 'processed_h5py', run), followlinks=True).next()[1]
            shuffle(segs_in_run)  # shuffle on each epoch to allow shuffle False

            run_labels = None
            try:
                run_labels = h5py.File(
                    os.path.join(data_folder_dir,
                                'processed_h5py',
                                 run,
                                 'run_labels.h5py'),
                    'r')
            except Exception:
                continue

            # Ignore invalid runs
            ignored = False
            for ignore in ignore_list:
                if ignore in run_labels and run_labels[ignore][0]:
                    ignored = True
                    break
            if ignored:
                continue

            ignored = len(require_one) > 0 
            for require in require_one:
                if require in run_labels and run_labels[require][0]:
                    ignored = False
                    break
            if ignored:
                continue

            print 'Loading Run ' + run
            for seg in segs_in_run:
                images = h5py.File(
                    os.path.join(
                        data_folder_dir,
                        'processed_h5py',
                        run,
                        seg,
                        'images.h5py'),
                    'r')

                metadata = h5py.File(
                    os.path.join(data_folder_dir,
                        'processed_h5py',
                         run,
                         seg,
                         'metadata.h5py'),
                    'r')


                length = len(images['left'])
                self.run_files.append({'images': images, 'metadata': metadata, 'run_labels' : run_labels})

                self.visible.append(self.total_length)  # visible indicies

                # invisible is not actually used at all, but is extremely useful
                # for debugging indexing problems and gives very little slowdown
                self.invisible.append(self.full_length + 7) # actual indicies mapped

                self.total_length += (length - (10 * stride - 1) - 7)
                self.full_length += length

        # Create row gradient
        self.row_gradient = torch.FloatTensor(94, 168)
        for row in range(94):
            self.row_gradient[row, :] = row / 93.

        # Create col gradient
        self.col_gradient = torch.FloatTensor(94, 168)
        for col in range(168):
            self.col_gradient[:, col] = col / 167.

        self.stride = stride

    def __getitem__(self, index):
        run_idx, t = self.create_map(index)

        list_camera_input = []

        list_camera_input.append(
            torch.from_numpy(
                self.run_files[
                    run_idx]['images']['left'][t - 7]))


        #Six timesteps?
        for delta_time in range(6, -1, -1):
            list_camera_input.append(
                torch.from_numpy(
                    self.run_files[
                        run_idx]['images']['left'][t - delta_time,:,:,1:2]))

        #only two timesteps?
        list_camera_input.append(
            torch.from_numpy(
                self.run_files[
                    run_idx]['images']['right'][t - 1,:,:,1:2]))

        list_camera_input.append(
            torch.from_numpy(
                self.run_files[
                    run_idx]['images']['right'][t,:,:,1:2]))

        camera_data = torch.cat(list_camera_input, 2)
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)

        final_camera_data = torch.FloatTensor(14, 94, 168)
        final_camera_data[0:12, :, :] = camera_data
        final_camera_data[12, :, :] = self.row_gradient
        final_camera_data[13, :, :] = self.col_gradient

        # Get behavioral mode
        metadata_raw = self.run_files[run_idx]['run_labels']
        metadata = torch.FloatTensor(128, 23, 41)
        metadata[:] = 0.
        for label_idx, cur_label in enumerate(['racing', 'follow', 'direct', 'play', 'furtive', 'clockwise', 'counterclockwise']):
            metadata[label_idx, :, :] = int(cur_label in metadata_raw and metadata_raw[cur_label][0])

        # Get Ground Truth
        steer = []
        motor = []

        for i in range(0, self.stride * 10, self.stride):
            steer.append(float(self.run_files[run_idx]['metadata']['steer'][t + i]))
        for i in range(0, self.stride * 10, self.stride):
            motor.append(float(self.run_files[run_idx]['metadata']['motor'][t + i]))
        for i in range(0, self.stride * 20, self.stride):
            motor.append(0.)
\
        final_ground_truth = torch.FloatTensor(steer + motor) / 99.

        mask = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # use all data
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # no mask
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # no mask
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # no mask

        return final_camera_data, metadata, final_ground_truth, mask

    def __len__(self):
        if self.max_len == -1:
            return self.total_length
        return min(self.total_length, self.max_len)

    def create_map(self, global_index):
        for idx, length in enumerate(self.visible[::-1]):
            if global_index >= length:
                return len(self.visible) - idx - 1, global_index - length + 7

if __name__ == '__main__':
    train_dataset = Dataset('/hostroot/data/dataset/bair_car_data_Main_Dataset', ['furtive'], [])
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=500,
                                                    shuffle=False, pin_memory=False)
    start = time.time()
    for cam, meta, truth, mask in train_data_loader:
        cur = time.time()
        print(500./(cur - start))
        start = cur
        pass

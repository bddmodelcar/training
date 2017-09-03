import numpy as np
import h5py
import torch
import torch.utils.data as data
import sys
from random import shuffle
import os
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, data_folder_dir, require_one, ignore_list, stride=10):
        self.runs = os.walk(os.path.join(data_folder_dir, 'h5py')).next()[1]
        self.run_files = []

        # Initialize List of Files
        # self.shuffle_runs()
        # self.runs.sort()
        self.run_list = []
        self.total_length = 0
        for run in self.runs:
            images = h5py.File(
                os.path.join(
                    data_folder_dir,
                    'h5py',
                    run,
                    'flip_images.h5py'),
                'r')

            metadata = h5py.File(
                os.path.join(data_folder_dir,
                             'h5py',
                             run,
                             'left_timestamp_metadata.h5py'),
                'r')

            run_labels = h5py.File(
                os.path.join(data_folder_dir,
                             'h5py',
                             run,
                             'run_labels.h5py'),
                'r')

            ignored = False
            for ignore in ignore_list:
                if ignore in run_labels and run_labels[ignore]:
                    ignored = True
                    break
            if ignored:
                continue

            ignored = len(require_one) > 0 
            for require in require_one:
                if require not in run_labels or run_labels[require]:
                    ignored = False
                    break
            if ignored:
                continue

            length = images['left_image_flip']['vals'].shape[0]
            self.run_files.append({'images': images, 'metadata': metadata, 'run_labels' : run_labels})
            self.run_list.append(
                self.total_length)  # Get rid of the first 7 frames as starting points
            self.total_length += (length - (10 * stride - 1) + 7)

        self.run_list = self.run_list[:-1]  # Get rid of last element (speed)

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
                    run_idx]['images']['left_image_flip']['vals'][
                        t - 7]))

        for delta_time in range(6, -1, -1):
            list_camera_input.append(
                torch.from_numpy(
                    self.run_files[
                        run_idx][
                            'images'][
                        'left_image_flip']['vals'][
                            t - delta_time,
                             :,
                             :,
                             1:2]))

        list_camera_input.append(
            torch.from_numpy(
                self.run_files[
                    run_idx][
                        'images'][
                    'right_image_flip']['vals'][
                        t - 1,
                         :,
                         :,
                         1:2]))

        list_camera_input.append(
            torch.from_numpy(
                self.run_files[
                    run_idx][
                        'images'][
                    'right_image_flip']['vals'][
                        t,
                         :,
                         :,
                         1:2]))

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
        metadata = torch.FloatTensor(20, 11, 20)
        metadata[:] = 0.
        for label_idx, cur_label in enumerate(['racing', 'follow', 'direct', 'play', 'furtive', 'clockwise', 'counterclockwise']):
            metadata[label_idx, :, :] = int(cur_label in metadata_raw and metadata_raw[cur_label][0])

        # Get Ground Truth
        steer = []
        motor = []

        for i in range(0, self.stride * 10, self.stride):
            steer.append(self.run_files[run_idx]['metadata']['steer'][t + i])
        for i in range(0, self.stride * 10, self.stride):
            motor.append(self.run_files[run_idx]['metadata']['motor'][t + i])

        final_ground_truth = torch.FloatTensor(steer + motor) / 99.

        return final_camera_data, metadata, final_ground_truth

    def __len__(self):
        return self.total_length

    def create_map(self, global_index):
        for idx, length in enumerate(self.run_list[::-1]):
            if global_index >= length:
                return len(self.run_list) - idx - 1, global_index - length + 7

    def shuffle_runs(self):
        shuffle(self.runs)

if __name__ == '__main__':
    train_dataset = Dataset('/hostroot/data/dataset/bair_car_data_Main_Dataset/',
                            [], [])
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=500,
                                                    shuffle=True, pin_memory=False)

    # print train_dataset.run_list
    # print train_dataset.create_map()
    for camera_data, metadata, ground_truth in train_data_loader:
        pass

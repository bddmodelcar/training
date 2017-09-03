import numpy as np
import h5py
import torch
import torch.utils.data as data
import sys
from random import shuffle
import os


class Dataset(data.Dataset):

    def __init__(self, data_folder_dir, require_one, ignore_list, stride=10):
        self.runs = os.walk(os.path.join(data_folder_dir, 'h5py')).next()[1]
        self.run_files = []

        # Initialize List of Files
        self.shuffle_runs()
        self.run_list = [-7]
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
                total_length -
                7)  # Get rid of the first 7 frames as starting points
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

    def __getitem__(self, index)
        run_idx, time_idx = self.create_map(index)

        list_camera_input = []
        list_camera_input.append(
            torch.from_numpy(
                run_files[
                    run_idx][
                        'images'][
                    'left_image_flip'][
                        t - 7]))

        for delta_time in range(6, -1, -1):
            list_camera_input.append(
                torch.from_numpy(
                    run_files[
                        run_idx][
                            'images'][
                        'left_image_flip'][
                            t - delta_time,
                             :,
                             :,
                             1:2]))

        list_camera_input.append(
            torch.from_numpy(
                run_files[
                    run_idx][
                        'images'][
                    'right_image_flip'][
                        t - 1,
                         :,
                         :,
                         1:2]))

        list_camera_input.append(
            torch.from_numpy(
                run_files[
                    run_idx][
                        'images'][
                    'right_image_flip'][
                        t,
                         :,
                         :,
                         1:2]))

        camera_data = torch.cat(list_camera_input, 2)
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)

        final_camera_data = torch.FloatTensor()
        final_camera_data[data_number, 0:12, :, :] = camera_data
        final_camera_data[data_number, 12, :, :] = self.row_gradient
        final_camera_data[data_number, 13, :, :] = self.col_gradient

        # Get Ground Truth
        steer = []
        motor = []

        for i in range(stride * 10, stride):
            steer.append(run_files[run_idx]['metadata']['steer'][t + i])
        for i in range(stride * 10, stride):
            motor.append(run_files[run_idx]['metadata']['motor'][t + i])

        final_ground_truth = torch.FloatTensor(steer + motor) / 99.

        return final_camera_data

    def create_map(self, global_index):
        for idx, length in enumerate(self.run_list[::-1]):
            if global_index >= length:
                return idx, global_index - length

    def shuffle_runs(self):
        shuffle(self.runs)

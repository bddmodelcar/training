import h5py
import torch
from torch.autograd import Variable
import torch.utils.data as data

class MergedDataset(data.Dataset):

    def __init__(self, hdf5_list, pre=False):
        self.datasets = []
        self.start = []
        self.end = []
        self.total_count = 0
        for f in hdf5_list:
           print f
           h5_file = h5py.File(f, 'r')
           self.datasets.append(h5_file)
           self.start.append(self.total_count)
           self.total_count += h5_file['train_camera_data'].shape[0]
           self.end.append(self.total_count)

    def __getitem__(self, index):
        for idx, lim in enumerate(self.end):
            if lim > index:
                datanum = idx
                break
        dataset = self.datasets[datanum]
        index -= self.start[datanum]
        camera_data = dataset['train_camera_data'][index, :, :, :]
        metadata = dataset['train_metadata'][index, :, :, :]
        target_data = dataset['train_target_data'][index, :]
        camera_data = camera_data.astype('float32') / 255. - 0.5
        metadata = metadata.astype('float32')
        target_data = target_data.astype('float32') / 99.
        return camera_data, metadata, target_data

    def __len__(self):
        return self.total_count

import h5py
import torch
import torch.utils.data as data
import sys

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

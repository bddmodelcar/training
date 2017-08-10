import h5py
import torch
import torch.utils.data as data

class HDF5Dataset(data.Dataset):

    def __init__(self, filename):
        self.h5file = h5py.File(filename, 'r')
        self.length = self.h5file['train_camera_data'].shape[0]

    def __getitem__(self, index):
        camera_data = self.h5file['train_camera_data'][index, :, :, :]
        metadata = self.h5file['train_metadata'][index, :, :, :]
        target_data = self.h5file['train_target_data'][index, :]
        return camera_data, metadata, target_data
    def __len__(self):
        return self.length

class MergedDataset(data.Dataset):

    def __init__(self, hdf5_list):
        self.datasets = []
	self.start = []
	self.end = []
        self.total_count = 0
        for f in hdf5_list:
           h5_file = h5py.File(f, 'r')
           dataset = (h5_file['train_camera_data'],
		      h5_file['train_metadata'],
		      h5_file['train_target_data'])
           self.datasets.append(dataset)
	   self.start.append(self.total_count)
           self.total_count += dataset[0].shape[0]
	   self.end.append(self.total_count)

    def __getitem__(self, index):
	for idx, lim in enumerate(self.end):
            if lim >= index:
                start_dataset = idx - 1
        index -= self.start[start_dataset] 

    def __len__(self):
      return self.total_count


  def __getitem__(self, index):
     
      dataset_index=-1
      #print 'index ',index
      for i in xrange(len(self.limits)-1,-1,-1):
        #print 'i ',i
        if index>=self.limits[i]:
          dataset_index=i
          break
      #print 'dataset_index ',dataset_index
      assert dataset_index>=0, 'negative chunk'

      in_dataset_index = index-self.limits[dataset_index]

      return self.datasets[dataset_index][in_dataset_index], self.datasets_gt[dataset_index][in_dataset_index]

  def __len__(self):

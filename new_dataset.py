import numpy as np
import h5py
import torch
import os
import torch.utils.data as data


# test our microphone but snapping and hearing if it does to different ears when played
# start and end index are for handling huge amounts of data when certain data needs to be accessed
# efficiently. For now, keep audio data inside car_info

class Dataset(data.Dataset):
    def __init__(self, data_dir, n_frames):
        self.data_dir = data_dir
        self.n_frames = n_frames
        self.moments = []   
        #self.left_imgs, self.right_imgs, self.steer_cmds, self.throttle_cmds = [], [], [], []
        self.dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.scale = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1).cuda()

        
        # print(self.sort_filelist(self.data_dir))
        for file_name in self.sort_filelist(self.data_dir):
            print(file_name)
            hdf5_file = h5py.File(self.data_dir + '/' + file_name, 'r') # find a more pythonic way to open up dir/file
            left_data = hdf5_file.get('left')
            right_data = hdf5_file.get('right')
            steer_data = hdf5_file.get('steer')
            throttle_data = hdf5_file.get('throttle')
            lefts, rights, steers, throttles= [], [], [], []
           
            for i in range(len(left_data)):
                lefts.append(left_data[i])
                rights.append(right_data[i])
                steers.append(steer_data[i])
                throttles.append(throttle_data[i])

                # every n_frames, create data moment
                if (i + 1) % self.n_frames == 0:
                   # print('DataMoment', len(lefts), len(rights), len(steers), len(throttles), i)
                    
                    if len(lefts) == self.n_frames:
                       # if(steers[0] > 90 or steers[3] > 90 or steers[6] > 90 or steers[9] > 90):
                        #    print(steers)
                        self.moments.append(DataMoment(
                                            lefts, 
                                            rights, 
                                            steers, 
                                            throttles, 
                                            None, 
                                            i, 
                                            self.n_frames))
                   
                    lefts, rights, steers, throttles= [], [], [], []



    def sort_filelist(self, data_dir):
        file_list = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith('hdf5'):
                file_list.append(file_name)
        return sorted(file_list)


    def __len__(self):
        return len(self.moments)

    def __getitem__(self, index):
        moment = self.moments[index]
        camera_data = []

        for i in range(self.n_frames):
            camera_data.append(torch.from_numpy(moment.left_imgs[i]).to('cuda:0')) # is it okay to use numpy instead of float?
            # changes .to(torch.device) to .to('cuda:0') bc the former wasn't working for unknown reason, but should not be hardcoded.
            camera_data.append(torch.from_numpy(moment.right_imgs[i]).to('cuda:0'))
           # print('img is what kind of tensor?:', type(camera_data[-1]), camera_data[-1].device)
            

        # maybe do the cuda part before this line, instead of in the normalization.
        camera_data = torch.cat(camera_data, 2)
        #print('camera_data is what kind of tensor?:', type(camera_data))
        #print('should be cuda...:',  camera_data.device)
        camera_data = camera_data.cuda().float() / 255. - 0.5 # maybe remove cuda?
        #print('camera_data should be cuda. Currently it is', camera_data.device)
        camera_data = torch.transpose(camera_data, 0, 2) 
        camera_data = torch.transpose(camera_data, 1, 2)
        #camera_data = camera_data.unsqueeze(0)
#        print(camera_data.size())
        camera_data = self.scale(camera_data)
        camera_data = self.scale(camera_data)

        all_steers = moment.steers
        double_steers = []
        for i in all_steers:
            double_steers.extend([i, i])

        double_steers = torch.Tensor(double_steers).to(self.dev)
        all_throttles = torch.Tensor(moment.throttles).to(self.dev)
        #print('Steers: {}').format(double_steers)
        #all_steers = all_steers.unsqueeze(0)        
        return camera_data, double_steers


class DataMoment():
    def __init__(self, left_imgs, right_imgs, steers, throttles, start_index=None, end_index=0, num_frames=1):
        self.left_imgs = left_imgs
        self.right_imgs = right_imgs
        self.steers = steers
        self.throttles = throttles
        self.end_index = end_index
        self.num_frames = num_frames
        if not start_index: # do I need start and end index??
            self.start_index = end_index - num_frames
        else:
            self.start_index = start_index

        # TODO: stack images in a moment, right?


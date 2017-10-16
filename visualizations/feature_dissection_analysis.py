import cv2
import argparse

from Parameters import ARGS
import torch
from torch.autograd import Variable

import libs
from libs import *
from nets import SqueezeNet

import torch.nn as nn
import torch.nn.functional as F

import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import h5py
import os

import threading

def convert_input_data_to_img(camera_data):
    temp_cam_data = camera_data.data.cpu().numpy()[0, :, :, :]
    xshape = temp_cam_data.shape[1]
    yshape = temp_cam_data.shape[2]
    n_frames = 10
    img = np.zeros((xshape * n_frames, yshape * 2, 3))
    ctr = 0
    
    for t in range(n_frames):
        for camera in ('left', 'right'):
            for c in range(3):
                if camera == 'left':
                    img[t * xshape:(t + 1) * xshape, 0:yshape, c] = temp_cam_data[ctr, :, :]
                elif camera == 'right':
                    img[t * xshape:(t + 1) * xshape, yshape:,  c] = temp_cam_data[ctr, :, :]
                ctr += 1
                
    return img

def convert_img_to_input_data(img):
    n_frames = 10
    n_cameras = 2
    xshape = img.shape[0]/n_frames
    yshape = img.shape[1]/n_cameras
    camera_data = np.zeros((n_frames*6, xshape, yshape))
    ctr = 0
    
    for t in range(n_frames):
        for camera in ('left', 'right'):
            for c in range(3):
                if camera == 'left':
                    camera_data[ctr, :, :] = img[t * xshape:(t + 1) * xshape, 0:yshape, c]
                elif camera == 'right':
                    camera_data[ctr, :, :] = img[t * xshape:(t + 1) * xshape, yshape:,  c]
                ctr += 1

    camera_data = torch.from_numpy(camera_data)
    camera_data = camera_data.cuda().float()/255. - 0.5
    camera_data = camera_data.unsqueeze(0)
    camera_data = Variable(camera_data)

    return camera_data
###
def laplacian_pyramid(pyr_size, level, output_vals):
    orig_pyr_vals = np.ones((pyr_size,pyr_size,3))*np.mean(output_vals.flatten())
    orig_pyr_vals[0:output_vals.shape[0], 0:output_vals.shape[1],:] = output_vals[:,:,:]
    pyr_vals = orig_pyr_vals.copy()
    
    diff = []
    for i in range(level): #range(int(np.log2(pyr_size))):
        orig_pyr_vals = pyr_vals.copy()
        pyr_vals = cv2.pyrDown(pyr_vals)

        if i + 1 == level:
            diff = orig_pyr_vals - cv2.pyrUp(pyr_vals)
            for j in range(i):
                diff = cv2.pyrUp(diff)
            
    return diff

###

ARGS.nframes = 10
net = SqueezeNet.SqueezeNet().cuda()
#model_path = '/home/bala/training/save/only_Tilden_no_Smyth/epoch15_save_0.002994.weights' #unstructured
model_path = '/home/bala/training/save/campus_local_no_Smyth_lr1/epoch10_save_0.001918.weights' #structured

save_data = torch.load(model_path)
net.load_state_dict(save_data)

raw_metadata = ['racing','caffe', 'follow', 'direct', 'play', 'furtive']

metadata = Variable(torch.randn(1, 6, 11, 20).type(torch.DoubleTensor).cuda())#old SN: Variable(torch.randn(1, 128, 23, 41).cuda())
    
all_layer_funcs = []
for i in range(6):  # custom size because i know the shape of this sequential                                                                                                                                         
    all_layer_funcs.append(net.pre_metadata_features[i])

all_layer_funcs.append(lambda x: torch.cat((x, metadata), 1))

for i in range(7):  # custom size because i know the shape of this sequential                                                                                                                                         
    all_layer_funcs.append(net.post_metadata_features[i])

all_layer_funcs.append(net.final_output)

all_layer_funcs.append(lambda x: x.view(x.size(0), -1))

###
runs_dir = '/home/dataset/bair_car_data/hdf5/runs/'
all_files = os.listdir(runs_dir)
num_files = len(all_files)

all_structured_files = []
all_unstructured_files = []
all_campus_files = []

for fname in all_files:
    f = h5py.File(runs_dir+fname, 'r') #keys are: 'labels', 'segments'
    if len(f.keys()) >= 2:
        if f['labels']['campus'][0] == 1:
            all_campus_files.append(fname)
            all_structured_files.append(fname)
        elif f['labels']['local'][0] == 1:
            all_structured_files.append(fname)
        elif f['labels']['Tilden'][0] == 1:
            all_unstructured_files.append(fname)
            
###
            
n_frames = 10
n_cameras = 2

layer_num = 3

runs_dir = '/home/dataset/bair_car_data/hdf5/runs/'

raw_metadata = ['racing','caffe', 'follow', 'direct', 'play', 'furtive']

ideal_data_moments_path = '/home/bala/training/visualizations/ideal_data_moments/'
ideal_data_moments_fnames = [fname for fname in os.listdir(ideal_data_moments_path) if 'moments1' in fname]

for ideal_data_moments_fname in [ideal_data_moments_fnames[0]]:#ideal_data_moments_fnames:
    ideal_data_moments = np.load(ideal_data_moments_path+ideal_data_moments_fname).item()

    all_data_moments_act_sums_by_dims = {}
    
    for dim_key in ideal_data_moments.keys():
        filename = ideal_data_moments[dim_key]['filename'] # flip_play_Nino_to_campus_08Oct16_09h00m00s_Mr_Blue_1c.hdf5'
        segment_num = ideal_data_moments[dim_key]['segment_num']
        timepoint_num = ideal_data_moments[dim_key]['timepoint_num']
        f = h5py.File(runs_dir+filename, 'r')
        
        l_frames = []
        r_frames = []
        speedup_factor = 5
        for frame_num in range(n_frames*speedup_factor):
            if timepoint_num+frame_num < f['segments'][segment_num]['left'].shape[0]:
                l_frames.append(f['segments'][segment_num]['left'][timepoint_num+frame_num])
                r_frames.append(f['segments'][segment_num]['right'][timepoint_num+frame_num])
        
        xshape = l_frames[0].shape[0]
        yshape = l_frames[0].shape[1]
        data_moment_img = np.zeros((xshape*n_frames, yshape*n_cameras, 3), np.uint8)
        for frame_num in range(n_frames):
            data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = l_frames[frame_num]
            data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = r_frames[frame_num]


        orig_data_moment_img = data_moment_img.copy()
            
        all_spatial_inputs = [data_moment_img.copy()]
        all_optical_flow_inputs = []
        all_stereo_inputs = [data_moment_img.copy()]

        ## optical flow
        # case 0: slow down speed by 1/infinity 
        temp_data_moment_img = data_moment_img.copy()
        
        for frame_num in range(10):
            temp_data_moment_img[frame_num*xshape:(frame_num + 1)*xshape, :,:] = data_moment_img[0:xshape, :,:]
            
        all_optical_flow_inputs.append(temp_data_moment_img.copy())

        # case 1: slow down speed by 1/5
        temp_data_moment_img = data_moment_img.copy()

        for frame_num in range(5):
            temp_data_moment_img[frame_num*xshape:(frame_num + 1)*xshape, :,:] = data_moment_img[0:xshape, :,:]
            temp_data_moment_img[(frame_num + 5)*xshape:(frame_num + 6)*xshape, :,:] = data_moment_img[xshape:2*xshape, :,:]
            
        all_optical_flow_inputs.append(temp_data_moment_img.copy())

        # case 2: slow down speed by 1/2
        temp_data_moment_img = data_moment_img.copy()

        for frame_num in range(5):
            temp_data_moment_img[2*frame_num*xshape:((2*frame_num)+1)*xshape, :,:] = data_moment_img[frame_num*xshape:(frame_num+1)*xshape, :,:]
            temp_data_moment_img[((2*frame_num) + 1)*xshape:((2*frame_num)+2)*xshape, :,:] = data_moment_img[frame_num*xshape:(frame_num+1)*xshape, :,:]
            
        all_optical_flow_inputs.append(temp_data_moment_img.copy())

        # case 3: normal speed
        all_optical_flow_inputs.append(orig_data_moment_img.copy())
        
        # case 4: speed up by 2x
        temp_data_moment_img = data_moment_img.copy()
        speedup = 2
        for frame_num in range(10):
            if frame_num*speedup < len(l_frames):
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = l_frames[frame_num*speedup]
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = r_frames[frame_num*speedup]
            else:
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = np.ones((xshape, yshape, 3))*np.nan
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = np.ones((xshape, yshape, 3))*np.nan
                
        all_optical_flow_inputs.append(temp_data_moment_img.copy())

        # case 5: speed up by 3x
        temp_data_moment_img = data_moment_img.copy()
        speedup = 3
        for frame_num in range(10):
            if frame_num*speedup < len(l_frames):
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = l_frames[frame_num*speedup]
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = r_frames[frame_num*speedup]
            else:
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = np.ones((xshape, yshape, 3))*np.nan
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = np.ones((xshape, yshape, 3))*np.nan

        all_optical_flow_inputs.append(temp_data_moment_img.copy())

        # case 6: speed up by 5x
        temp_data_moment_img = data_moment_img.copy()
        speedup = 5
        for frame_num in range(10):
            if frame_num*speedup < len(l_frames):
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = l_frames[frame_num*speedup]
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = r_frames[frame_num*speedup]
            else:
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = np.ones((xshape, yshape, 3))*np.nan
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = np.ones((xshape, yshape, 3))*np.nan
                
        all_optical_flow_inputs.append(temp_data_moment_img.copy())
                                                                                                                                                                                                                    
        ## spatial flow
        if layer_num == 3: # if first Fire layer
            pyr_size = 256

        orig_data_moment_power = np.sum(np.square(data_moment_img.flatten()))

        for spatial_level in range(1,9):
            temp_data_moment_img = np.zeros((xshape*n_frames, yshape*n_cameras, 3), np.uint8)
            for frame_num in range(n_frames):
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = laplacian_pyramid(pyr_size, spatial_level, l_frames[frame_num])[0:xshape,0:yshape]
                temp_data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = laplacian_pyramid(pyr_size, spatial_level, r_frames[frame_num])[0:xshape,0:yshape]
                
            new_data_moment_power = np.sum(np.square(temp_data_moment_img.flatten()))
            
            temp_data_moment_img = orig_data_moment_power*temp_data_moment_img/new_data_moment_power
            temp_data_moment_img = temp_data_moment_img.astype(np.uint8)
            
            all_spatial_inputs.append(temp_data_moment_img.copy())


        cond_act_sums_by_dims = {}
        
        for data_moment_img in all_optical_flow_inputs:
            camera_data = convert_img_to_input_data(data_moment_img) #Variable(255 * torch.randn(1, 12, 94, 168).
            metadata = torch.FloatTensor()
            for mode in raw_metadata:
                if mode == 'caffe':
                    metadata = torch.cat((torch.FloatTensor(1, 11, 20).fill_(0), metadata), 0)
                else:
                    metadata = torch.cat((torch.FloatTensor(1, 11, 20).fill_(f['labels'][mode][0]), metadata), 0)
                    
            output = camera_data
            layer_to_vis = layer_num
            visualized_layer = None
            for i, layer in enumerate(all_layer_funcs):
                output = layer(output)
                if i == layer_to_vis:
                    #print output.size()
                    visualized_layer = layer
                    break    
            
            output_vals = np.array(output.data.cpu().numpy())
            output_vals = output_vals.reshape(output_vals.shape[1], output_vals.shape[2], output_vals.shape[3])
            #output_vals = output_vals[dim_key,:,:]
            
            dim_act_sums = np.sum(np.sum(np.abs(output_vals), axis=1), axis=1)

            for dim in range(len(dim_act_sums)):
                if not (dim in cond_act_sums_by_dims.keys()):
                    cond_act_sums_by_dims[dim] = []
                cond_act_sums_by_dims[dim].append(dim_act_sums[dim])

        for dim in cond_act_sums_by_dims.keys():
            if not (dim in all_data_moments_act_sums_by_dims.keys()):
                all_data_moments_act_sums_by_dims[dim] = []
            all_data_moments_act_sums_by_dims[dim].append(cond_act_sums_by_dims[dim])


num_feats = 7
f = plt.figure()    
num_dim_sqrt = int(np.ceil(np.sqrt(128)))
    
for dim in all_data_moments_act_sums_by_dims.keys():
    ax1 = f.add_subplot(num_dim_sqrt, num_dim_sqrt, dim+1)

    for moment_ind in range(len(all_data_moments_act_sums_by_dims)):
        moment_ind_acts = all_data_moments_act_sums_by_dims[dim][moment_ind]
        
        ax1.plot(range(-3,num_feats-3), moment_ind_acts, 'k-', alpha=.05)
    
    '''
    ## Bar graph
    mean_act_sums_per_cond = np.nanmean(all_data_moments_act_sums_by_dims[dim], axis=0)
    std_act_sums_per_cond = np.nanstd(all_data_moments_act_sums_by_dims[dim], axis=0)
    
    plt.bar(range(-3,num_feats-3), mean_act_sums_per_cond, yerr=std_act_sums_per_cond/np.ceil(np.sqrt(128)))
    '''
    
plt.show()    

###

'''
num_feats = 9.0
num_rows = 1

f = plt.figure()
for i in range(int(num_feats)):
    f.add_subplot(num_rows,int(np.ceil(num_feats/num_rows)),i+1)
    plt.imshow(all_spatial_inputs[i])

plt.show()
'''
###
'''                
runs_dir = '/home/dataset/bair_car_data/hdf5/runs/'
all_files = os.listdir(runs_dir)
num_files = len(all_files)

all_structured_files = []
all_unstructured_files = []
all_campus_files = []

for fname in all_files:
    f = h5py.File(runs_dir+fname, 'r') #keys are: 'labels', 'segments'
    if len(f.keys()) >= 2:
        if f['labels']['campus'][0] == 1:
            all_campus_files.append(fname)
            all_structured_files.append(fname)
        elif f['labels']['local'][0] == 1:
            all_structured_files.append(fname)
        elif f['labels']['Tilden'][0] == 1:
            all_unstructured_files.append(fname)

layer_num = 0
file_num = 0
filename = all_files[0]
segment_num = '0'
timepoint_num = 0

n_frames = 10
n_cameras = 2

max_dim_act_sums1 = np.array([])
max_data_moment_info1 = {} # keys: dim, subkeys: fname, segment, timepoint
all_acts_img_real1 = None

max_dim_act_sums2 = np.array([])
max_data_moment_info2 = {} # keys: dim, subkeys: fname, segment, timepoint 
all_acts_img_real2 = None

num_dims_to_vis = None; sqrt_num_dims = None
out_xshape = None; out_yshape = None
outputs_min = 0; outputs_max = 255 ## BALA: change this when get stereo vals


img_upsample_coeff = 2

for filename in all_structured_files:
    f = h5py.File(runs_dir+filename, 'r') #keys are: 'labels', 'segments'

    for segment_num in f['segments'].keys():
        for timepoint_num in range(np.array(f['segments'][segment_num]['left']).shape[0] - n_frames + 1):
            
            l_frames = []
            r_frames = []

            for frame_num in range(n_frames):
                l_frames.append(f['segments'][segment_num]['left'][timepoint_num+frame_num])
                r_frames.append(f['segments'][segment_num]['right'][timepoint_num+frame_num])

            xshape = l_frames[0].shape[0]
            yshape = l_frames[0].shape[1]

            data_moment_img = np.zeros((xshape*n_frames, yshape*n_cameras, 3), np.uint8)

            for frame_num in range(n_frames):
                data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = l_frames[frame_num]
                data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = r_frames[frame_num]


            camera_data = convert_img_to_input_data(data_moment_img) #Variable(255 * torch.randn(1, 12, 94, 168).cuda())
            metadata = torch.FloatTensor()
            for mode in raw_metadata:
                if mode == 'caffe':
                    metadata = torch.cat((torch.FloatTensor(1, 11, 20).fill_(0), metadata), 0)
                else:
                    metadata = torch.cat((torch.FloatTensor(1, 11, 20).fill_(f['labels'][mode][0]), metadata), 0)
                
                    
            output = camera_data
            layer_to_vis = layer_num
            visualized_layer = None
            for i, layer in enumerate(all_layer_funcs):
                output = layer(output)
                if i == layer_to_vis:
                    #print output.size()
                    visualized_layer = layer
                    break

            

            output_vals = np.array(output.data.cpu().numpy())
            output_vals = output_vals.reshape(output_vals.shape[1], output_vals.shape[2], output_vals.shape[3])

            dim_act_sums1 = np.abs(np.sum(np.sum(output_vals, axis=1), axis=1))
            dim_act_sums2 = np.sum(np.sum(np.abs(output_vals), axis=1), axis=1)

            if len(max_dim_act_sums1) == 0:
                max_dim_act_sums1 = dim_act_sums1.copy()
                max_dim_act_sums2 = dim_act_sums2.copy()
                
                num_dims_to_vis = output_vals.shape[0]
                sqrt_num_dims = int(np.ceil(np.sqrt(num_dims_to_vis)))                
                out_xshape = output_vals.shape[1]; out_yshape = output_vals.shape[2]
                
                all_acts_img_real1 = np.zeros((sqrt_num_dims*out_xshape, sqrt_num_dims*out_yshape)).astype(np.uint8)
                all_acts_img_real2 = np.zeros((sqrt_num_dims*out_xshape, sqrt_num_dims*out_yshape)).astype(np.uint8)
                
            for dim in range(dim_act_sums1.shape[0]):
                if dim_act_sums1[dim] >= max_dim_act_sums1[dim]:
                    max_dim_act_sums1[dim] = dim_act_sums1[dim]
                    max_data_moment_info1[dim] = {'filename': filename,
                                                 'segment_num': segment_num,
                                                 'timepoint_num': timepoint_num}
                    

                    dim_i_outputs_real = output_vals[dim,:,:]
                    dim_i_outputs_real -= np.nanmin(dim_i_outputs_real.flatten())
                    dim_i_outputs_real /= np.nanmax(dim_i_outputs_real.flatten())
                    dim_i_outputs_real *= 255
                    
                    all_acts_img_real1[(dim/sqrt_num_dims)*out_xshape:((dim/sqrt_num_dims)+1)*out_xshape,
                                       (dim%sqrt_num_dims)*out_yshape:((dim%sqrt_num_dims)+1)*out_yshape] = dim_i_outputs_real

                if dim_act_sums2[dim] >= max_dim_act_sums2[dim]:
                    max_dim_act_sums2[dim] = dim_act_sums2[dim]
                    max_data_moment_info2[dim] = {'filename': filename,
                                                  'segment_num': segment_num,
                                                  'timepoint_num': timepoint_num}
                    
                    dim_i_outputs_real = output_vals[dim,:,:]
                    dim_i_outputs_real -= np.nanmin(dim_i_outputs_real.flatten())
                    dim_i_outputs_real /= np.nanmax(dim_i_outputs_real.flatten())
                    dim_i_outputs_real *= 255
                    all_acts_img_real2[(dim/sqrt_num_dims)*out_xshape:((dim/sqrt_num_dims)+1)*out_xshape,
                                       (dim%sqrt_num_dims)*out_yshape:((dim%sqrt_num_dims)+1)*out_yshape] = dim_i_outputs_real

np.save('layer0_structured_ideal_data_moments1.npy', max_data_moment_info1)
np.save('layer0_structured_ideal_data_moments2.npy', max_data_moment_info2)
'''
'''

        filename = main_window_params['filename'] #all_files[file_num]
    
        f = h5py.File(runs_dir+filename, 'r') #keys are: 'labels', 'segments'
        num_segments = main_window_params['num_segments']#len(np.sort(f['segments'].keys()))
        num_timepoints = main_window_params['num_timepoints']#f['segments']['0']['left'].shape[0]

        #segment_num = str(cv2.getTrackbarPos('Data File Segment:', 'main_window'))
        #timepoint_num = cv2.getTrackbarPos('Data File Segment Timepoint:', 'main_window')
    
        n_frames = 10
        l_frames = []
        r_frames = []
        
        for frame_num in range(n_frames):
            l_frames.append(f['segments'][main_window_params['segment_num']]['left'][main_window_params['timepoint_num']+frame_num])
            r_frames.append(f['segments'][main_window_params['segment_num']]['right'][main_window_params['timepoint_num']+frame_num])

        xshape = l_frames[0].shape[0]
        yshape = l_frames[0].shape[1]
        n_cameras = 2
        data_moment_img = np.zeros((xshape*n_frames, yshape*n_cameras, 3), np.uint8)
        
        for frame_num in range(n_frames):
            data_moment_img[frame_num*xshape:(frame_num+1)*xshape, 0:yshape, :] = l_frames[frame_num]
            data_moment_img[frame_num*xshape:(frame_num+1)*xshape, yshape:, :] = r_frames[frame_num]

        if not (main_window_params['stereo_control_num'] == 0):
            if main_window_params['stereo_control_num'] == 1:
                data_moment_img[:,yshape:,:] = data_moment_img[:,0:yshape,:] # sets left to right side
            elif main_window_params['stereo_control_num'] == 2:
                data_moment_img[:,0:yshape,:] = data_moment_img[:,yshape:,:] # sets right to left side

        if not (main_window_params['rgb_control_num'] == 0):
            if main_window_params['rgb_control_num'] == 1:
                data_moment_img[:,:,1] = 0#data_moment_img[:,:,0] # sets g+b channels to r
                data_moment_img[:,:,2] = 0#data_moment_img[:,:,0]
            elif main_window_params['rgb_control_num'] == 2:
                data_moment_img[:,:,0] = 0#data_moment_img[:,:,1] # sets r+b channels to g
                data_moment_img[:,:,2] = 0#data_moment_img[:,:,1]
            elif main_window_params['rgb_control_num'] == 3:
                data_moment_img[:,:,0] = 0#data_moment_img[:,:,2] # sets r+g channels to b
                data_moment_img[:,:,1] = 0#data_moment_img[:,:,2]

        if not (main_window_params['temporal_control_num'] == 0):
            if main_window_params['temporal_control_num'] == 1:
                for frame_num in range(4):
                    data_moment_img[(frame_num+1)*xshape:(frame_num+2)*xshape,:,:] = data_moment_img[0:xshape,:,:]

                    data_moment_img[(frame_num+6)*xshape:(frame_num+7)*xshape,:,:] = data_moment_img[5*xshape:6*xshape,:,:]

            if main_window_params['temporal_control_num'] == 2:
                for frame_num in range(5):
                    f_ind = frame_num*2
                    data_moment_img[(f_ind+1)*xshape:(f_ind+2)*xshape,:,:] = data_moment_img[(f_ind)*xshape:(f_ind+1)*xshape,:,:]
                
        camera_data = convert_img_to_input_data(data_moment_img) #Variable(255 * torch.randn(1, 12, 94, 168).cuda())

        
        output = camera_data
        layer_to_vis = main_window_params['layer_num']
        visualized_layer = None
        for i, layer in enumerate(all_layer_funcs):
            output = layer(output)
            if i == layer_to_vis:
                visualized_layer = layer
                break
            
        output_vals = np.array(output.data.cpu().numpy())
        output_vals = output_vals.reshape(output_vals.shape[1], output_vals.shape[2], output_vals.shape[3])
        dim_act_sums = np.abs(np.sum(np.sum(output_vals, axis=1), axis=1))
        top_dim_act_inds = np.argsort(dim_act_sums)[::-1] # reversed so it's max to min

        num_dims_to_vis = output_vals.shape[0]
        sqrt_num_dims = int(np.ceil(np.sqrt(num_dims_to_vis)))
        out_xshape = output_vals.shape[1]; out_yshape = output_vals.shape[2]
        
        all_acts_img_sorted = np.zeros((sqrt_num_dims*out_xshape, sqrt_num_dims*out_yshape)).astype(np.uint8)
        all_acts_img_real = np.zeros((sqrt_num_dims*out_xshape, sqrt_num_dims*out_yshape)).astype(np.uint8)

        out_padding = pad_buff*conv_output_dims
        in_padding = pad_buff*conv_input_dims
        conv_weights1 = .5*np.ones((conv_output_dims*max_kernel_size*upsample_coeff+out_padding,
                                    conv_input_dims*max_kernel_size*upsample_coeff+in_padding))
        conv_weights2 = .5*np.ones((conv_output_dims*max_kernel_size*upsample_coeff+out_padding,
                                    conv_input_dims*max_kernel_size*upsample_coeff+in_padding))
        conv_weights3 = .5*np.ones((conv_output_dims*max_kernel_size*upsample_coeff+out_padding,
                                    conv_input_dims*max_kernel_size*upsample_coeff+in_padding))

        outputs_min = np.nanmin(output_vals.flatten())
        outputs_max = np.nanmax(output_vals.flatten())
        
        for i in range(num_dims_to_vis):            
            dim_i_outputs_sorted = output_vals[top_dim_act_inds[i],:,:]
            dim_i_outputs_sorted -= outputs_min ##np.nanmin(dim_i_outputs_sorted.flatten())
            dim_i_outputs_sorted /= outputs_max #np.nanmax(dim_i_outputs_sorted.flatten())
            dim_i_outputs_sorted *= 255

            dim_i_outputs_real = output_vals[i,:,:]
            dim_i_outputs_real -= outputs_min #np.nanmin(dim_i_outputs_real.flatten())
            dim_i_outputs_real /= outputs_max #np.nanmax(dim_i_outputs_real.flatten())
            dim_i_outputs_real *= 255
            
            all_acts_img_sorted[(i/sqrt_num_dims)*out_xshape:((i/sqrt_num_dims)+1)*out_xshape,
                                (i%sqrt_num_dims)*out_yshape:((i%sqrt_num_dims)+1)*out_yshape] = dim_i_outputs_sorted

            all_acts_img_real[(i/sqrt_num_dims)*out_xshape:((i/sqrt_num_dims)+1)*out_xshape,
                              (i%sqrt_num_dims)*out_yshape:((i%sqrt_num_dims)+1)*out_yshape] = dim_i_outputs_real

        if isinstance(visualized_layer,nn.Conv2d):
            weights = visualized_layer.weight.cpu().data.numpy()
            for i in range(conv_output_dims):
                for j in range(conv_input_dims):
                    if i < weights.shape[0] and j < weights.shape[1]:
                        weight_ij_filter = weights[i,j,:,:].repeat(upsample_coeff, axis=0).repeat(upsample_coeff, axis=1)
                        conv_weights1[i*pad_buff + i*max_kernel_size*upsample_coeff:i*pad_buff + (i+1)*max_kernel_size*upsample_coeff,
                                      j*pad_buff + j*max_kernel_size*upsample_coeff:j*pad_buff + (j+1)*max_kernel_size*upsample_coeff] = weight_ij_filter

        elif isinstance(visualized_layer,SqueezeNet.Fire):
            all_acts_img_sorted = all_acts_img_sorted.repeat(img_upsample_coeff, axis=0).repeat(img_upsample_coeff, axis=1)
            all_acts_img_real = all_acts_img_real.repeat(img_upsample_coeff, axis=0).repeat(img_upsample_coeff, axis=1)
            
            squeeze_weights = visualized_layer.squeeze.weight.cpu().data.numpy()
            expand1_weights = visualized_layer.expand1x1.weight.cpu().data.numpy()
            expand3_weights = visualized_layer.expand3x3.weight.cpu().data.numpy()
            for i in range(conv_output_dims):
                for j in range(conv_input_dims):
                    squeeze_weight_ij_filter = squeeze_weights[i,j,:,:].repeat(max_kernel_size*upsample_coeff, axis=0).repeat(max_kernel_size*upsample_coeff, axis=1)
                    expand1_weight_ij_filter = expand1_weights[i,j,:,:].repeat(max_kernel_size*upsample_coeff, axis=0).repeat(max_kernel_size*upsample_coeff, axis=1)
                    expand3_weight_ij_filter = expand3_weights[i,j,:,:].repeat(upsample_coeff, axis=0).repeat(upsample_coeff, axis=1)
                    
                    conv_weights1[i*pad_buff + i*max_kernel_size*upsample_coeff:i*pad_buff + (i+1)*max_kernel_size*upsample_coeff,
                                  j*pad_buff + j*max_kernel_size*upsample_coeff:j*pad_buff + (j+1)*max_kernel_size*upsample_coeff] = squeeze_weight_ij_filter

                    conv_weights2[i*pad_buff + i*max_kernel_size*upsample_coeff:i*pad_buff + (i+1)*max_kernel_size*upsample_coeff,
                                  j*pad_buff + j*max_kernel_size*upsample_coeff:j*pad_buff + (j+1)*max_kernel_size*upsample_coeff] = expand1_weight_ij_filter

                    conv_weights3[i*pad_buff + i*max_kernel_size*upsample_coeff:i*pad_buff + (i+1)*max_kernel_size*upsample_coeff,
                                  j*pad_buff + j*max_kernel_size*upsample_coeff:j*pad_buff + (j+1)*max_kernel_size*upsample_coeff] = expand3_weight_ij_filter
                    
        #print('layer_num: {}'.format(layer_num))
        #print('file_num: {}'.format(file_num))
        #print('segment_num: {}'.format(segment_num))
        #print('timepoint_num: {}'.format(timepoint_num))
        

cv2.destroyAllWindows()
'''

'''
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

dim_act_sums1 = []; dim_act_sums2 = []
num_feats = 4

for i in range(num_feats):
    dim_act_sums1.append(np.load('dim_act_sums1_stereo{}.npy'.format(i)))
    dim_act_sums2.append(np.load('dim_act_sums2_stereo{}.npy'.format(i)))

num_dim_sqrt = int(np.ceil(np.sqrt(128)))

f = plt.figure()

for dim in range(128):
    f.add_subplot(num_dim_sqrt, num_dim_sqrt, dim+1)
    dim_vals = []
    for i in range(num_feats):
        dim_vals.append(dim_act_sums2[i][dim])
    plt.bar(range(num_feats), dim_vals)

plt.show()
'''
'''

'''
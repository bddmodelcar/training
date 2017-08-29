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
    camera_data = np.zeros((1, n_frames*6, xshape, yshape))
    ctr = 0
    
    for t in range(n_frames):
        for camera in ('left', 'right'):
            for c in range(3):
                if camera == 'left':
                    camera_data[0, ctr, :, :] = img[t * xshape:(t + 1) * xshape, 0:yshape, c]
                elif camera == 'right':
                    camera_data[0, ctr, :, :] = img[t * xshape:(t + 1) * xshape, yshape:,  c]
                ctr += 1

    input_data = Variable(torch.from_numpy(camera_data).cuda())
    return input_data

def nothing(x):
    pass

new_main_window_params = {'files_type': None,
                          'filename': '',
                          'num_layers': 0,
                          'layer_num': 0,
                          'num_files': 0,
                          'file_num': 0,
                          'num_segments': 0,
                          'segment_num': '0',
                          'num_timepoints': 0,
                          'timepoint_num': 0,
                          'stereo_control_num': 0,
                          'rgb_control_num': 0,
                          'temporal_control_num': 0}

main_window_params = new_main_window_params.copy()

def restart_main_window():
    cv2.destroyWindow('main_window')
    for key in main_window_params.keys():
        main_window_params[key] = new_main_window_params[key]
        
    data_file_label = 'Data File' #main_window_params['filename']
    cv2.namedWindow('main_window')
    cv2.createTrackbar('Network Layer:','main_window', 0, main_window_params['num_layers'], change_layer_num)
    cv2.createTrackbar('Files Type Num:','main_window', 0, 3, change_files_type_num)
    cv2.createTrackbar('Data File Number:','main_window', 0, main_window_params['num_files']-1, change_file_num)
    cv2.createTrackbar('Data File Segment:','main_window', 0, main_window_params['num_segments']-1, change_segment_num)
    cv2.createTrackbar('Data File Segment Timepoint:','main_window', 0, main_window_params['num_timepoints']-10, change_timepoint_num)

    cv2.createTrackbar('Show Stereo Control Num:','main_window', 0, 2, change_stereo_control_num)
    cv2.createTrackbar('Show RGB Control Num:','main_window', 0, 3, change_rgb_control_num)
    cv2.createTrackbar('Show Temporal Control Num:','main_window', 0, 2, change_temporal_control_num)
    return

def change_layer_num(x):    
    main_window_params['layer_num'] = x
    new_main_window_params['layer_num'] = x
    return

def change_files_type_num(x):
    files_to_choose = None
    if x == 0: # all
        files_to_choose = all_files
    if x == 1: # smythe
        files_to_choose = all_structued_files
    if x == 2: # campus
        files_to_choose = all_unstructured_files
    if x == 3: # tilden
        files_to_choose = all_campus_files

    new_main_window_params['files_type'] = files_to_choose
    num_files = len(files_to_choose)
    new_main_window_params['num_files'] = num_files
    change_file_num(0)
    return

def change_file_num(x):
    new_main_window_params['files_type'] = main_window_params['files_type']
    new_main_window_params['segment_num'] = '0' #reset this so things don't break when window params reset
    new_main_window_params['timepoint_num'] = 0
    
    new_main_window_params['file_num'] = x
    filename = new_main_window_params['files_type'][new_main_window_params['file_num']]
    new_main_window_params['filename'] = filename
    
    f = h5py.File(runs_dir+filename, 'r') #keys are: 'labels', 'segments'
    new_main_window_params['num_segments'] = len(np.sort(f['segments'].keys()))
    new_main_window_params['num_timepoints'] = f['segments']['0']['left'].shape[0] #this works because segment_num gets defaulted to 0
    return

def change_segment_num(x):
    new_main_window_params['timepoint_num'] = 0 #reset this so things don't break when window params reset
    change_file_num(main_window_params['file_num'])

    filename = main_window_params['filename']
    f = h5py.File(runs_dir+filename, 'r') #keys are: 'labels', 'segments'
    
    new_main_window_params['segment_num'] = str(x)
    new_main_window_params['num_timepoints'] = f['segments'][str(x)]['left'].shape[0]
    return

def change_timepoint_num(x):
    main_window_params['timepoint_num'] = x
    return

def change_stereo_control_num(x):
    main_window_params['stereo_control_num'] = x

def change_rgb_control_num(x):
    main_window_params['rgb_control_num'] = x

def change_temporal_control_num(x):
    main_window_params['temporal_control_num'] = x

###

runs_dir = '/home/dataset/bair_car_data/hdf5/runs/'
all_files = os.listdir(runs_dir)
num_files = len(all_files)
main_window_params['num_files'] = num_files
main_window_params['files_type'] = all_files

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
filename = main_window_params['files_type'][file_num]
segment_num = '0'
timepoint_num = 0

filename = main_window_params['files_type'][main_window_params['file_num']]
main_window_params['filename'] = filename

f = h5py.File(runs_dir+filename, 'r') #keys are: 'labels', 'segments'
main_window_params['num_segments'] = len(np.sort(f['segments'].keys()))
main_window_params['num_timepoints'] = f['segments']['0']['left'].shape[0]

for key in new_main_window_params.keys():
    new_main_window_params[key] = main_window_params[key]

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

###

ARGS.nframes = 10
net = SqueezeNet.SqueezeNet().cuda()
model_path = '/home/bala/training/save/campus_local_no_Smyth/epoch05_save_0.001967.weights' #'/home/bala/pytorch_models/epoch6goodnet'
save_data = torch.load(model_path)
net.load_state_dict(save_data)

metadata = Variable(torch.randn(1, 6, 11, 20).type(torch.DoubleTensor).cuda())#old SN: Variable(torch.randn(1, 128, 23, 41).cuda())
all_layer_funcs = []
for i in range(6):  # custom size because i know the shape of this sequential
    all_layer_funcs.append(net.pre_metadata_features[i])

all_layer_funcs.append(lambda x: torch.cat((x, metadata), 1))

for i in range(7):  # custom size because i know the shape of this sequential
    all_layer_funcs.append(net.post_metadata_features[i])
    
all_layer_funcs.append(net.final_output)

all_layer_funcs.append(lambda x: x.view(x.size(0), -1))

camera_data = convert_img_to_input_data(data_moment_img) #Variable(255 * torch.randn(1, 12, 94, 168).cuda())
output = camera_data
layer_to_vis = main_window_params['layer_num']
visualized_layer = None
for i, layer in enumerate(all_layer_funcs):
    output = layer(output)
    if i == layer_to_vis:
        #print output.size()
        visualized_layer = layer
        break

output_vals = np.array(output.data.cpu().numpy())
output_vals = output_vals.reshape(output_vals.shape[1], output_vals.shape[2], output_vals.shape[3])

num_dims_to_vis = output_vals.shape[0]
sqrt_num_dims = int(np.ceil(np.sqrt(num_dims_to_vis)))
out_xshape = output_vals.shape[1]; out_yshape =output_vals.shape[2]
img_upsample_coeff = 2

all_acts_img_sorted = np.zeros((sqrt_num_dims*out_xshape, sqrt_num_dims*out_yshape)).astype(np.uint8)
all_acts_img_real = np.zeros((sqrt_num_dims*out_xshape, sqrt_num_dims*out_yshape)).astype(np.uint8)
    
main_window_params['num_layers'] = len(all_layer_funcs)
new_main_window_params['num_layers'] = len(all_layer_funcs)

###

conv_input_dims = 16
conv_output_dims = 16
max_kernel_size = 3
upsample_coeff = 8
pad_buff = 6
conv_weights1 = np.zeros((conv_output_dims*max_kernel_size*upsample_coeff, conv_input_dims*max_kernel_size*upsample_coeff))
conv_weights2 = np.zeros((conv_output_dims*max_kernel_size*upsample_coeff, conv_input_dims*max_kernel_size*upsample_coeff))
conv_weights3 = np.zeros((conv_output_dims*max_kernel_size*upsample_coeff, conv_input_dims*max_kernel_size*upsample_coeff))





###

cv2.namedWindow('main_window')
cv2.namedWindow('data_moment_window')
cv2.namedWindow('weights_window1')
cv2.namedWindow('weights_window2')
cv2.namedWindow('weights_window3')
cv2.namedWindow('acts_window_sorted')
cv2.namedWindow('acts_window_real')

data_file_label = 'Data File' #main_window_params['filename']
cv2.createTrackbar('Network Layer:','main_window', 0, main_window_params['num_layers'], change_layer_num)
cv2.createTrackbar('Files Type Num:','main_window', 0, 3, change_files_type_num)
cv2.createTrackbar('Data File Number:','main_window', 0, main_window_params['num_files']-1, change_file_num)
cv2.createTrackbar('Data File Segment:','main_window', 0, main_window_params['num_segments']-1, change_segment_num)
cv2.createTrackbar('Data File Segment Timepoint:','main_window', 0, main_window_params['num_timepoints']-10, change_timepoint_num)

cv2.createTrackbar('Show Stereo Control Num:','main_window', 0, 2, change_stereo_control_num)
cv2.createTrackbar('Show RGB Control Num:','main_window', 0, 3, change_rgb_control_num)
cv2.createTrackbar('Show Temporal Control Num:','main_window', 0, 2, change_temporal_control_num)



while(1):
    cv2.imshow('data_moment_window', data_moment_img)
    cv2.imshow('acts_window_sorted', all_acts_img_sorted)
    cv2.imshow('acts_window_real', all_acts_img_real)
    cv2.imshow('weights_window1', conv_weights1)
    cv2.imshow('weights_window2', conv_weights2)
    cv2.imshow('weights_window3', conv_weights3)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        restart_main_window()
        #break
    else:    
        #layer_num = cv2.getTrackbarPos('Network Layer:','main_window')
        #file_num = cv2.getTrackbarPos('Data File:', 'main_window')
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
        for i in range(num_dims_to_vis):            
            dim_i_outputs_sorted = output_vals[top_dim_act_inds[i],:,:]
            dim_i_outputs_sorted -= np.min(dim_i_outputs_sorted.flatten())
            dim_i_outputs_sorted /= np.max(dim_i_outputs_sorted.flatten())
            dim_i_outputs_sorted *= 255

            dim_i_outputs_real = output_vals[i,:,:]
            dim_i_outputs_real -= np.min(dim_i_outputs_real.flatten())
            dim_i_outputs_real /= np.max(dim_i_outputs_real.flatten())
            dim_i_outputs_real *= 255
            
            all_acts_img_sorted[(i/sqrt_num_dims)*out_xshape:((i/sqrt_num_dims)+1)*out_xshape,
                                (i%sqrt_num_dims)*out_yshape:((i%sqrt_num_dims)+1)*out_yshape] = dim_i_outputs_sorted

            all_acts_img_real[(i/sqrt_num_dims)*out_xshape:((i/sqrt_num_dims)+1)*out_xshape,
                              (i%sqrt_num_dims)*out_yshape:((i%sqrt_num_dims)+1)*out_yshape] = dim_i_outputs_real

            out_padding = pad_buff*conv_output_dims
            in_padding = pad_buff*conv_input_dims
            conv_weights1 = .5*np.ones((conv_output_dims*max_kernel_size*upsample_coeff+out_padding,
                                      conv_input_dims*max_kernel_size*upsample_coeff+in_padding))
            conv_weights2 = .5*np.ones((conv_output_dims*max_kernel_size*upsample_coeff+out_padding,
                                      conv_input_dims*max_kernel_size*upsample_coeff+in_padding))
            conv_weights3 = .5*np.ones((conv_output_dims*max_kernel_size*upsample_coeff+out_padding,
                                      conv_input_dims*max_kernel_size*upsample_coeff+in_padding))

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

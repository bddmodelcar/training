"""Data preprocessing code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Data
import Batch
import Utils

import matplotlib.pyplot as plt

import torch
import h5py


def main():
    ARGS.batch_size = 1
    # Set Up PyTorch Environment
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(ARGS.gpu)
    torch.cuda.device(ARGS.gpu)

    data = Data.Data()
    batch = Batch.Batch()
    rate_counter = Utils.RateCounter()

    h5File = h5py.File("/data/tpankaj/preprocess_follow.hdf5")
    train_len = 9708866 # replace this with true count
    val_len = 91749 # replace this with true count
    #train_camera_data = h5File.create_dataset("train_camera_data", (train_len, 12, 94, 168), dtype='uint8')
    #train_metadata = h5File.create_dataset("train_metadata", (train_len, 128, 23, 41), dtype='uint8')
    #train_target_data = h5File.create_dataset("train_target_data", (train_len, 20), dtype='uint8')
    val_camera_data = h5File.create_dataset("val_camera_data", (val_len, 12, 94, 168), dtype='uint8')
    val_metadata = h5File.create_dataset("val_metadata", (val_len, 128, 23, 41), dtype='uint8')
    val_target_data = h5File.create_dataset("val_target_data", (val_len, 20), dtype='uint8')

    # Save training data
    #count = 0
    #while not data.train_index.epoch_complete:  # Epoch of training
    #    camera_data, metadata, target_data = batch.fill(data, data.train_index)
    #    count += 1
    #    #print("Start: " + str(data.train_index.ctr - ARGS.batch_size))
    #    #print("End: " + str(data.train_index.ctr))
    #    train_camera_data[count - ARGS.batch_size : count, :, :, :] = camera_data.cpu().numpy()
    #    train_metadata[count - ARGS.batch_size : count, :, :, :] = metadata.cpu().numpy()
    #    train_target_data[count - ARGS.batch_size : count, :] = target_data.cpu().numpy()
    #    rate_counter.step()
    #print(count)
    
    # Save validation data
    val_count = 0
    while not data.val_index.epoch_complete:  # Epoch of validation
        camera_data, metadata, target_data = batch.fill(data, data.val_index)
        val_count += 1
        #print("Start: " + str(data.train_index.ctr - ARGS.batch_size))
        #print("End: " + str(data.train_index.ctr))
        val_camera_data[val_count - ARGS.batch_size : val_count, :, :, :] = camera_data.cpu().numpy()
        val_metadata[val_count - ARGS.batch_size : val_count, :, :, :] = metadata.cpu().numpy()
        val_target_data[val_count - ARGS.batch_size : val_count, :] = target_data.cpu().numpy()
        rate_counter.step()
    print(count)
    print(val_count)

if __name__ == '__main__':
    main()

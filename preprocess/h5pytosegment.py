import numpy as np
import h5py
import os

prefix = '/hostroot/data/dataset/bair_car_data_Main_Dataset/h5py/direct_local_19Dec16_16h00m00s_Mr_Orange'

f_meta = h5py.File(os.path.join(prefix, 'left_timestamp_metadata.h5py'), 'r')
f_img = h5py.File(os.path.join(prefix, 'flip_images.h5py'), 'r')

rounded_state = np.round(f_meta['state'][:]) # TODO: export to h5py
for i in range(1, len(rounded_state) - 1):
    if not rounded_state[i - 1] == 4 and not rounded_state[i + 1] == 4 and rounded_state[i] == 4:
        rounded_state[i] = rounded_state[i - 1]
        
consecutive_seq_idx = np.zeros(len(f_meta['ts']))

def is_valid_timestamp(state, motor, allow_state = [1, 2, 3, 5, 7], min_motor = 53):
    return state in allow_state and motor > min_motor

for i in range(1, len(consecutive_seq_idx)):
    consecutive_seq_idx[i] = int(is_valid_timestamp(rounded_state[i], f_meta['motor'][i]) and f_meta['ts'][i] - f_meta['ts'][i-1] < 0.3)

# find closest right idx to each left idx (in time)
left_idx_to_right = []
left_ts = f_img['left_image_flip']['ts'][:]
right_ts = f_img['right_image_flip']['ts'][:]
for i in range(len(left_ts)):
    diffs = np.abs(left_ts[i] - right_ts[max(0, i - 10) : min(i + 10, len(left_ts) - 1)])
    left_idx_to_right.append(np.argmin(diffs) + max(0, i - 10))


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

condition = consecutive_seq_idx.astype(bool)

# Print the start and stop indicies of each region where the absolute 
# values of x are below 1, and the min and max of each of these regions
for start, stop in contiguous_regions(condition):
    if stop - start > 180:
        print start, stop
    # segment = x[start:stop]
    # print segment.min(), segment.max()

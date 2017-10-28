from __future__ import print_function, unicode_literals

import os
import pickle
from multiprocessing import Pool

import h5py
import numpy as np


def process(run_name):

    input_prefix = os.path.join(
        '/hostroot/data/dataset/bair_car_data_new_28April2017/h5py',
        run_name,
    )
    output_prefix = os.path.join(
        '/hostroot/data/dataset/bair_car_data_new_28April2017/processed_h5py',
        run_name,
    )

    os.makedirs(output_prefix, exist_ok=True)

    f_meta = h5py.File(
        os.path.join(input_prefix, 'left_timestamp_metadata.h5py'),
        'r',
    )
    f_img = h5py.File(
        os.path.join(input_prefix, 'flip_images.h5py'),
        'r',
    )
    f_normal_img = h5py.File(
        os.path.join(input_prefix, 'original_timestamp_data.h5py'),
        'r',
    )

    rounded_state = np.round(f_meta['state'][:])  # TODO: export to h5py
    for i in range(1, len(rounded_state) - 1):
        if not (rounded_state[i - 1] == 4 or rounded_state[i + 1] == 4) and rounded_state[i] == 4:
            rounded_state[i] = rounded_state[i - 1]

    consecutive_seq_idx = np.zeros(len(f_meta['ts']))

    def is_valid_timestamp(state, motor, allow_state=[1, 2, 3, 5, 7], min_motor=53):
        return state in allow_state and motor > min_motor

    aruco = None
    # Trick to get easy tuple unpacking.
    steer = 4 * [np.zeros(len(f_meta['ts']))]

    ccdirectsteer, ccwdirectsteer, ccfollowsteer, ccwfollowsteer = steer

    try:
        aruco = pickle.load(
            open(
                '/hostroot/data/dataset/Aruco_Steering_Trajectories/pkl/' + run_name + '.pkl', 'r'
            )
        )
    except Exception:
        return
    for i in range(1, len(consecutive_seq_idx)):
        consecutive_seq_idx[i] = int(
            is_valid_timestamp(rounded_state[i], f_meta['motor'][i])
            and f_meta['ts'][i] - f_meta['ts'][i - 1] < 0.3
        )
        if f_meta['ts'][i] not in aruco['Direct_Arena_Potential_Field'][0]:
            consecutive_seq_idx[i] = 0
        else:
            ccdirectsteer[i] = int(
                aruco['Direct_Arena_Potential_Field'][0][f_meta['ts'][i]]['steer']
            )
            ccwdirectsteer[i] = int(
                aruco['Direct_Arena_Potential_Field'][1][f_meta['ts'][i]]['steer']
            )
            ccfollowsteer[i] = int(
                aruco['Follow_Arena_Potential_Field'][0][f_meta['ts'][i]]['steer']
            )
            ccwfollowsteer[i] = int(
                aruco['Follow_Arena_Potential_Field'][1][f_meta['ts'][i]]['steer']
            )
    # find closest right idx to each left idx (in time)
    left_idx_to_right = []
    left_ts = f_img['left_image_flip']['ts'][:]
    right_ts = f_img['right_image_flip']['ts'][:]
    for i in range(len(left_ts)):
        try:
            diffs = np.abs(left_ts[i] - right_ts[max(0, i - 10):min(i + 10, len(left_ts) - 1)])
            # print diffs
            # print right_ts[max(0, i - 10) : min(i + 10, len(left_ts) - 1)]
            left_idx_to_right.append(np.argmin(diffs) + max(0, i - 10))
        except Exception:
            consecutive_seq_idx[i] = 0  # if there is a problem get rid of it

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
            idx = np.r_[idx, condition.size]  # Edit

        # Reshape the result into two columns
        idx.shape = (-1, 2)
        return idx

    condition = consecutive_seq_idx.astype(bool)

    def save_h5py(
        state,
        motor,
        steer,
        left,
        right,
        time,
        seg_num,
        seg_length,
    ):
        output_dir = os.path.join(output_prefix, run_name, "seg_" + str(seg_num))
        os.makedirs(output_dir)

        new_f_images = h5py.File(os.path.join(output_dir, "images.h5py"))

        new_f_images.create_dataset('left', (seg_length, 94, 168, 3), dtype='uint8')
        new_f_images['left'][:] = left

        new_f_images.create_dataset('right', (seg_length, 94, 168, 3), dtype='uint8')
        new_f_images['right'][:] = right

        new_f_images.create_dataset('ts', (seg_length, ), dtype='int')
        new_f_images['ts'][:] = time

        new_f_metadata = h5py.File(os.path.join(output_dir, "metadata.h5py"))

        for i, mode in enumerate('cwdirect', 'ccwdirect', 'cwfollow', 'ccwfollow'):
            new_f_metadata.create_dataset(mode, (seg_length, ), dtype='uint8')
            new_f_metadata[mode][:] = steer[i].astype('uint8')

        new_f_metadata.create_dataset('motor', (seg_length, ), dtype='uint8')
        new_f_metadata['motor'][:] = motor.astype('uint8')

        new_f_metadata.create_dataset('state', (seg_length, ), dtype='uint8')
        new_f_metadata['state'][:] = state.astype('uint8')

    # Print the start and stop indices of each region where the absolute
    # values of x are below 1, and the min and max of each of these regions
    seg_num = 0
    for start, stop in contiguous_regions(condition):
        if stop - start > 10:
            state = rounded_state[start:stop]
            motor = f_meta['motor'][start:stop]
            steer = [99 - s[start:stop] for s in steer]

            left = np.zeros((stop - start, 94, 168, 3), dtype='uint8')
            right = f_img['left_image_flip']['vals'][start:stop]  # notice the flip
            for count, i in enumerate(range(start, stop)):
                left[count] = f_img['right_image_flip']['vals'][left_idx_to_right[i]]

            time = np.array(list(range(len(left))))
            seg_length = len(left)

            save_h5py(state, motor, steer, left, right, time, seg_num, seg_length)

            seg_num += 1

            # Unflipped Images
            steer = (
                ccdirectsteer[start:stop],
                ccwdirectsteer[start:stop],
                ccfollowsteer[start:stop],
                ccwfollowsteer[start:stop],
            )
            time = np.arange(len(left))

            right = np.zeros((stop - start, 94, 168, 3), dtype='uint8')
            left = f_normal_img['left_image']['vals'][start:stop]
            for count, i in enumerate(range(start, stop)):
                right[count] = f_normal_img['right_image']['vals'][left_idx_to_right[i]]
            time = np.array(list(range(len(left))))

            save_h5py(state, motor, steer, left, right, time, seg_num, seg_length)

            seg_num += 1

            print(start, stop)


if __name__ == '__main__':
    input_prefix = '/hostroot/data/dataset/bair_car_data_new_28April2017/h5py/'
    run_names = next(os.walk(input_prefix))[1]
    pool = Pool(processes=10)
    pool.map(process, run_names)
    # process(run_names[0])

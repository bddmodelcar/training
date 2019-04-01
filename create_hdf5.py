import rosbag
import numpy as np
import h5py
import os
from enum import Enum
from scipy.io.wavfile import write


class Data(Enum):

    LEFT = 0
    RIGHT = 1
    TIME = 2
    STATE = 3
    STEER = 4
    THROTTLE = 5




def image_to_numpy(msg):
    channels = 3
    shape = (msg.height, msg.width, channels)
    data = np.frombuffer(msg.data, dtype='uint8')
    data = np.resize(data, shape)
    return data




def audio_to_numpy(msg):
    arr = np.fromstring(msg.data, dtype=audio_type)
    
    # I planned to resize it but when saving a multi-dimensional np array in H5PY
    # dataset with variable_length dtype, it seems to truncate dimensions higher than 1-D
    #arr = np.resize(arr, (-1, 2)) # Here you can set 2 channels to denote audio is stereo. (Hasn't made audio playback successful for me though)
    
    return arr




directory = '/home/nitzan/test_bag_audio'
#directory = '/home/nitzan/test_bag'
#directory = '/home/nitzan/2-25-19_with_mic'
data_len = len(Data) # num of items in enum Data
audio = True # Record audio in HDF5 file
audio_type = 'int16' # Set if binary-format audio data should be read as uint8 or int16

for rosbag_file in os.listdir(directory):
    if not rosbag_file.endswith('.bag'):
        continue
    data = dict()
    left_data = []
    right_data = []
    state_data = []
    steer_data = []
    throttle_data = []
    timestamp_data = []
    audio_data = []
    audio_ts_data = []
    first_img = -1


    with rosbag.Bag(directory + '/' + rosbag_file, 'r') as bag:

        num_left = bag.get_message_count('/zed/left/image_rect_color')
        num_right = bag.get_message_count('/zed/right/image_rect_color')
        num_imgs = num_left if num_left < num_right else num_right

        # this code relies on /car_info to contain msg.imgNum which matches
        # an image's sequence num, and thus allows syncing of metadata to img
        for topic, msg, time in bag.read_messages():
            if topic == '/zed/left/image_rect_color':
                if first_img < 0:
                    first_img = msg.header.seq
                    print('Starting image set [' + str(first_img) + ' - ' + str(first_img+num_imgs) + ']', rosbag_file)
                img_id = msg.header.seq
                if img_id not in data:
                    data[img_id] = [None] * data_len
                data[img_id][Data.LEFT.value] = image_to_numpy(msg)
                data[img_id][Data.TIME.value] = time.to_sec()

            elif topic == '/zed/right/image_rect_color':
                img_id = msg.header.seq
                if img_id not in data:
                    data[img_id] = [None] * data_len
                data[img_id][Data.RIGHT.value] = image_to_numpy(msg)

            elif topic == '/car_info':
                img_id = msg.imgNum
                if img_id not in data:
                    data[img_id] = [None] * data_len
                data[img_id][Data.STATE.value] = msg.state
                data[img_id][Data.STEER.value] = msg.steer
                data[img_id][Data.THROTTLE.value] = msg.throttle
            
            elif topic == '/audio/audio' and audio and first_img >= 0: # start recording only once images have begun being recorded
                audio_data.append(audio_to_numpy(msg)) # audio data is raw bytes. Most likely each signal is 16 bits per ear (for stereo audio). If not 16, then audio would be unsigned 8 bits per ear. I don't know the sample rate for sure but could be 16,000.
                audio_ts_data.append(time.to_sec())

    
        # If any image didn't get accompanying metadata, then copy over its neighbor's data.
        # Some images don't get metadata because of ROS messages being unsynced
        for key, value in data.items():
            if not value[Data.STATE.value]:
                if (key+1) in data.keys():
                    neighbor_key = key + 1
                elif (key-1) in data.keys():
                    neighbor_key = key - 1
                data[key][Data.STATE.value] = data[neighbor_key][Data.STATE.value]
                data[key][Data.STEER.value] = data[neighbor_key][Data.STEER.value]
                data[key][Data.THROTTLE.value] = data[neighbor_key][Data.THROTTLE.value]

      
        # Aggregates data into lists of left, right, steer, throttle, and audio
        # only append data to lists if all data values are present, and state == 2 == human annotation.
        # Also, state != None implies that steer and throttle are also != None
        for i in range(first_img, first_img + num_imgs):
            try:
                data_elem = data[i]
            except KeyError as e:
                pass # an image is missing for some reason
            state = data_elem[Data.STATE.value]
            left = data_elem[Data.LEFT.value]
            right = data_elem[Data.RIGHT.value]
            if state and state == 2:
                # Looks unpythonic but it's the only way to check if an np.ndarray is None
                if left is None:
                    continue
                if right is None:
                    continue
                left_data.append(left)
                right_data.append(right)
                steer_data.append(data_elem[Data.STEER.value])
                throttle_data.append(data_elem[Data.THROTTLE.value])
                timestamp_data.append(data_elem[Data.TIME.value])


        # convert lists to np.ndarray
        left_data = np.array(left_data)
        right_data = np.array(right_data)
        steer_data = np.array(steer_data)
        throttle_data = np.array(throttle_data)
        timestamp_data = np.array(timestamp_data)
        audio_data = np.array(audio_data)
        audio_ts_data = np.array(audio_ts_data)

    # Create datasets with numpy arrays as the data
    # Sometimes there isn't data to record if state was never == 2, so checking if left_data is empty.
    if len(left_data) > 0:
        # save HDF5 file
        f = h5py.File(os.path.splitext(directory + '/' + rosbag_file)[0] + '.h5', 'w')
        f.create_dataset('timestamp', data=timestamp_data)
        f.create_dataset('left', data=left_data)
        f.create_dataset('right', data=right_data)
        f.create_dataset('steer', data=steer_data)
        f.create_dataset('throttle', data=throttle_data)
        if audio:
            if len(audio_data) == 0:
                print('Was told to record audio, but no audio found in this rosbag')
            else:
                dt = h5py.special_dtype(vlen=np.dtype(audio_type))
                f.create_dataset('audio', data=audio_data, dtype=dt)
                f.create_dataset('audio_ts', data=audio_ts_data)
        print('Done')
    else:
        print('No data recorded in state #2 (Data collection mode) in this rosbag. No HDF5 file created.')

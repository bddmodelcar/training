from libs.vis2 import *
import libs.Segment_Data as Segment_Data
from Parameters import args


class DataIndex:
    """
    Index object, keeps track of position in data stack.
    """

    def __init__(self, valid_data_moments, ctr, epoch_counter):
        self.valid_data_moments = valid_data_moments
        self.ctr = ctr
        self.epoch_counter = epoch_counter
        self.epoch_complete = False


class Data:
    def __init__(self):

        # Load hdf5 segment data
        self.hdf5_runs_path = self.hdf5_segment_metadata_path = args.data_path
        self.hdf5_runs_path += '/hdf5/runs'
        self.hdf5_segment_metadata_path += '/hdf5/segment_metadata'

        Segment_Data.load_Segment_Data(self.hdf5_segment_metadata_path,
                                       self.hdf5_runs_path)

        # Load data indexes for training and validation
        print('loading train_valid_data_moments...')
        self.train_index = DataIndex(lo(opjD('train_all_steer')), -1, 0)
        print('loading val_valid_data_moments...')
        self.val_index = DataIndex(lo(opjD('val_all_steer')), -1, 0)

    @staticmethod
    def get_data(run_code, seg_num, offset):
        data = Segment_Data.get_data(run_code, seg_num, offset,
                                     args.stride * args.nsteps, offset,
                                     args.nframes, ignore=args.ignore,
                                     require_one=args.require_one,
                                     use_states=args.use_states)
        return data

    def next(self, data_index):
        if data_index.ctr >= len(data_index.valid_data_moments):
            data_index.ctr = -1
            data_index.epoch_counter += 1
            data_index.epoch_complete = True
        if data_index.ctr == -1:
            data_index.ctr = 0
            print('shuffle start')
            random.shuffle(data_index.valid_data_moments)
            print('shuffle finished')
        data_index.ctr += 1
        return data_index.valid_data_moments[data_index.ctr]

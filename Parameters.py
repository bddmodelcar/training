import argparse  # default python library for command line argument parsing

parser = argparse.ArgumentParser(description='Train DNNs on model car data.',
                                 formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default=0, type=int, help='Cuda GPU ID')
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--no-display', dest='display', action='store_false')
parser.set_defaults(display=True)

parser.add_argument('--verbose', default=True, type=bool,
                    help='Debugging mode')
parser.add_argument('--aruco', default=True, type=bool, help='Use Aruco data')
parser.add_argument('--data-path', default='/home/karlzipser/Desktop/' +
                    'bair_car_data_Main_Dataset', type=str)
parser.add_argument('--resume-path', default=None, type=str, help='Path to' +
                    ' resume file containing network state dictionary')
parser.add_argument('--save-path', default='save', type=str, help='Path to' +
                    ' folder to save net state dictionaries.')

# nargs='+' allows for multiple arguments and stores arguments in a list
parser.add_argument('--ignore', default=('reject_run', 'left', 'out1_in2'),
                    type=str, nargs='+', help='Skips these labels in data.')
parser.add_argument('--require-one', default=(), type=str, nargs='+',
                    help='Skips data without these labels in data.')
parser.add_argument('--use-states', default=(1, 3, 5, 6, 7), type=str,
                    nargs='+', help='Skips data outside of these states.')

parser.add_argument('--nframes', default=2, type=int,
                    help='# timesteps of camera input')
parser.add_argument('--nsteps', default=10, type=int,
                    help='# of steps of time to predict in the future')
parser.add_argument('--stride', default=3, type=int,
                    help="number of timesteps between network predictions")

parser.add_argument('--save-time', default=60*30, type=int,
                    help='time to wait before saving network (seconds)')
parser.add_argument('--print-time', default=5, type=int,
                    help='time to wait before displaying network input/output')
parser.add_argument('--mini-train-time', default=60*30, type=int,
                    help='time to train before validating')
parser.add_argument('--mini-val-time', default=60*3, type=int,
                    help='time to validate before training')
parser.add_argument('--loss-timer', default=60/2, type=int,
                    help='interval over which loss is computed, seconds')

args = parser.parse_args()

"""Command line arguments parser configuration."""
import argparse  # default python library for command line argument parsing

PARSER = argparse.ArgumentParser(description='Train DNNs on model car data.',
                                 formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)

PARSER.add_argument('--gpu', default=0, type=int, help='Cuda GPU ID')
PARSER.add_argument('--batch-size', default=100, type=int)
PARSER.add_argument('--display', dest='display', action='store_true')
PARSER.add_argument('--no-display', dest='display', action='store_false')
PARSER.set_defaults(display=True)

PARSER.add_argument('--verbose', default=True, type=bool,
                    help='Debugging mode')
PARSER.add_argument('--aruco', default=True, type=bool, help='Use Aruco data')
PARSER.add_argument('--data-path', default='/home/karlzipser/Desktop/' +
                    'bair_car_data_Main_Dataset', type=str)
PARSER.add_argument('--resume-path', default=None, type=str, help='Path to' +
                    ' resume file containing network state dictionary')
PARSER.add_argument('--save-path', default='save', type=str, help='Path to' +
                    ' folder to save net state dictionaries.')

# nargs='+' allows for multiple arguments and stores arguments in a list
PARSER.add_argument('--ignore', default=('reject_run', 'left', 'out1_in2'),
                    type=str, nargs='+', help='Skips these labels in data.')
PARSER.add_argument('--require-one', default=(), type=str, nargs='+',
                    help='Skips data without these labels in data.')
PARSER.add_argument('--use-states', default=(1, 3, 5, 6, 7), type=str,
                    nargs='+', help='Skips data outside of these states.')

PARSER.add_argument('--nframes', default=2, type=int,
                    help='# timesteps of camera input')
PARSER.add_argument('--nsteps', default=10, type=int,
                    help='# of steps of time to predict in the future')
PARSER.add_argument('--stride', default=3, type=int,
                    help="number of timesteps between network predictions")

PARSER.add_argument('--print-moments', default=1000, type=int,
                    help='# of moments between printing stats')

args = PARSER.parse_args()

import json
from Parameters import ARGS

def get_config(ARGS=ARGS, config_file_name=None):
    """
    Takes in command-line arguments as well as the configuration file and parses them.

    Parameters
    -----------------
    :param config_dict: <dict>: a dictionary of configuration settings to use
    :param ARGS: <key-val>: the command-line arguments to take, structured like a dictionary
    :param config_file_name: <str>: relative filepath to config file
    """
    config_file_name = './configs/' + (config_file_name or (ARGS and ARGS.config) or '')
    config_dict = json.load(open(config_file_name, 'r'))
    return config_dict


config = get_config()
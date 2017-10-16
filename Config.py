import json
from Parameters import ARGS

class Config(dict):
    def __init__(self, init_dict=None, ARGS=ARGS, config_file_name=None):
        """
        Takes in command-line arguments as well as the configuration file and parses them.

        Parameters
        -----------------
        :param init_dict: <dict>: a dictionary of configuration settings to use
        :param ARGS: <key-val>: the command-line arguments to take, structured like a dictionary
        :param config_file_name: <str>: relative filepath to config file
        """
        if init_dict:
            self.load_config(init_dict)
        config_file_name = './configs/' + (config_file_name or (ARGS and ARGS.config) or '')
        config_file_name.replace('./configs/configs/', './configs')
        if ARGS:
            ARGS.config = ''
        if config_file_name != './configs/':
            config_dict = json.load(open(config_file_name, 'r'))
            base_config = json.load(open('./configs/default.json', 'r'))
            self.load_config(base_config)
            self.load_config(config_dict, ARGS)

    def load_config(self, kv, ARGS=None):
        """
        Loads in a configuration from a dictionary

        :param kv: <key-val>: a key-val data structure
        :return: None
        """
        for key in kv:
            if key == 'gpu' and ARGS and ARGS.gpu != -1:
                self['gpu'] = ARGS
            if type(kv[key]) is dict:
                if key in self:
                    self[key].load_config(kv[key])
                else:
                    self[key] = Config(init_dict=kv[key], ARGS=ARGS, config_file_name=None)
            else:
             self[key] = kv[key]

config = Config()

import json
from Parameters import parse

class Config(dict):
    def __init__(self, init_dict=None, parser=parse, config_file_name=None):
        """
        Takes in command-line arguments as well as the configuration file and parses them.

        Parameters
        -----------------
        :param init_dict: <dict>: a dictionary of configuration settings to use
        :param config_file_name: <str>: relative filepath to config file
        """

        ARGS = parse()
        if init_dict:
            self.load_config(init_dict)
        config_file_name = './configs/' + (config_file_name or (ARGS and ARGS.config) or '')
        config_file_name = config_file_name.replace('configs/configs', 'configs')
        if ARGS:
            ARGS.config = ''
        if config_file_name != './configs/':
            self.load_helper(config_file_name, ARGS)

    def load_helper(self, path=None, ARGS=None):
        if path is None:
            base_config = json.load(open('./configs/default.json', 'r'))
            self.load_config(base_config)
        else:
            config_dict = json.load(open(path, 'r'))
            if config_dict['parent_config']:
                self.load_helper(('./configs/' + config_dict['parent_config'])
                                 .replace('./configs/configs/', './configs/'))
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

class Constants:
    def __init__(self, *args, **kwargs):
        self.n_frames = 2
        self.n_steps = 10
        self.train_data_loc = ''
        self.val_data_loc = ''
        self.train_batch_size = 250
        self.val_batch_size = 250

        own_vars = vars(self)
        for key in kwargs.keys():
            if key in own_vars.keys():
                own_vars[key] = kwargs[key]


squeezenet_constants = Constants()
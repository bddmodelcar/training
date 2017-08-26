"""Script to count the number of parameters in a PyTorch state_dict"""
import sys
import numpy as np
import torch

MODEL = torch.load(sys.argv[1])
params = 0 # pylint: disable=invalid-name
for key in MODEL:
    params += np.multiply.reduce(MODEL[key].shape)
print('Total number of parameters: ' + str(params))

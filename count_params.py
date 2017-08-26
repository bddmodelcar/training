import sys
import numpy as np
import torch

model = torch.load(sys.argv[1])
params = 0
for key in model:
    params += np.multiply.reduce(model[key].shape)
print('Total number of parameters: ' + str(params))

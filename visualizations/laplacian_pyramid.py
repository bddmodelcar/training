import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2

def laplacian_pyramid(level, output_vals):
    pyr_size = 64
    orig_pyr_vals = np.ones((pyr_size,pyr_size))*np.mean(output_vals.flatten())
    orig_pyr_vals[0:output_vals.shape[0], 0:output_vals.shape[1]] = output_vals[:,:]
    pyr_vals = orig_pyr_vals.copy()

    f = plt.figure()
    diffs = []
    for i in range(level):
        orig_pyr_vals = pyr_vals.copy()

        pyr_vals = cv2.pyrDown(pyr_vals)
        #pyr_vals = cv2.pyrUp(pyr_vals)
        
        diff = orig_pyr_vals - cv2.pyrUp(pyr_vals)

        for j in range(i):
            diff = cv2.pyrUp(diff)
        diffs.append(diff.copy())
            
        
        f.add_subplot(2,3,i+1)
        plt.imshow(diff[0:output_vals.shape[0], 0:output_vals.shape[1]])
    plt.show()
                                                    
#laplacian_pyramid(6, output_vals[0,:,:])

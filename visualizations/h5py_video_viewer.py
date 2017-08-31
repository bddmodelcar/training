import numpy as np
import h5py
from scipy import signal


f = h5py.File('/home/dataset/data_2017_08_29/bdd_aruco_demo/h5py/Mr_Lt_Blue_16_50_29Aug2017/flip_images.h5py', 'r')
# show an image
img = f['left_image_flip']['vals'][0, :, :, :]

import cv2
import numpy as np

def nothing(x):
    pass

#cv2.namedWindow('image', CV_WINDOW_NORMAL)
cv2.namedWindow('image', cv2.cv.CV_WINDOW_NORMAL)

# create trackbars for color change
cv2.createTrackbar('Timestep','image', 0, f['left_image_flip']['vals'].shape[0] - 1, nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    timestep = cv2.getTrackbarPos('Timestep','image')
    img = f['left_image_flip']['vals'][timestep, :, :, :]

cv2.destroyAllWindows()

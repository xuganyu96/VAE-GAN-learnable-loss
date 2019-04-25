import mxnet as mx
from mxnet import nd, image as mximage
import matplotlib.pyplot as plt
import numpy as np
import os

# This script is intended to be run from within the data processing script
# Hence all paths are relative to the main directory

# Specify the directory where the images are stored
IMG_DIR = './data/anime_faces/'
COLOR_CHANNELS = 'RGB'

# List the names of the files
img_filenames = os.listdir(IMG_DIR)
img_paths = [IMG_DIR + filename for filename in img_filenames]

# Read the first image to determine the shape
dummy_img_array = mximage.imread(img_paths[0], 
                                 flag = int(COLOR_CHANNELS == 'RGB'))
# Because the imread() method will return NDArray object of
# the shape (width, height, n_channels)
# we will extract these dimensionalities and create a
# 0 NDArray accordingly
width, height, n_channels = dummy_img_array.shape
# Finally, the number of samples can be extracted from the length of
# the path list
sample_size = len(img_paths)

# Initialize a zero NDArray to hold all the images
output = nd.zeros((sample_size, n_channels, width, height))

# Iterate through all the paths
for i in range(sample_size):
    img_path = img_paths[i]
    
    # Get the image NDArray; note that at this point
    # img_array is of shape (width, height, n_channels)
    # and of values 0 - 255
    # so we need to divide by 255 to get a probabilistic representation
    # and reshape it to be of the shape
    # (n_channels, width, height)
    img_array = mximage.imread(img_path, 
                               flag = int(COLOR_CHANNELS == 'RGB'))
    img_array = img_array.astype(np.float32) / 255.
    
    
    output[i] = img_array.reshape((n_channels, width, height))
    
    if (i+1) % 100 == 0:
        print(str(i+1) + '/' + str(sample_size) + ' processed')
        
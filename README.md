# Convolutions
Modify images using convolutional neural networks

Based on DeepDream https://github.com/google/deepdream greatly simplified.

Code for modifying images for Gina Chiao's (http://ginachiao.com) interactive art projects.

Dependencies:

  Python

  CUDA (if you want to use the GPU)
  
  Caffe with Python Caffe
  
  Numpy
  
  Scipy
  
  PIL
  
  It's easiest to build all this on Ubuntu 14.04
  
  
As of October 30 2015 it takes the image in the main directory, modifies it, and saves it in the frames directory.  Changing the image name, netwrok layer, network model, number of steps etc. requires editing the python files.


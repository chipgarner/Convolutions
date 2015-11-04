# imports
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
#import cv2
#from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# call the following to run Caffe operations on the GPU.
def use_gpu():
    caffe.set_mode_gpu()
    caffe.set_device(0) # select GPU device if multiple devices exist
	
def load_model():
    model_path = '../caffe-master/models/bvlc_googlenet/' # substitute your path here
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))
    
    net = caffe.Classifier('tmp.prototxt', param_fn,
                        mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                        channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
                        
    return net

def load_image(image_relative_path):
    img = np.float32(PIL.Image.open(image_relative_path))
    return img

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])
    
def saveResult(net, index, end, vis):
    # adjust image contrast and clip
    vis = vis*(255.0/np.percentile(vis, 99.98))     
    vis = np.uint8(np.clip(vis, 0, 255))

    pimg = PIL.Image.fromarray(vis)
       
    f = "frames/frame" + str(index) + end.replace('/', '-') + ".jpg"
    pimg.save(f, 'jpeg')
    print f
     
def showResult(net, vis):
    # adjust image contrast and clip
    vis = vis*(255.0/np.percentile(vis, 99.98))     
    vis = np.uint8(np.clip(vis, 0, 255))

    pimg = PIL.Image.fromarray(vis)
       
    pimg.show()

	
###########
def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, end, step_size=1.5, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g 
###################################################

def deepdream(net, base_img, iter_n=10, end='inception_4e/output', **step_params):
    # reshape and load image
    source = net.blobs['data']
    h, w, c = base_img.shape[:]
    source.reshape(1,3,h,w)
    source.data[0] = preprocess(net, base_img)

    for i in xrange(iter_n):
        make_step(net, end=end, **step_params)
            
        # visualization
        vis = deprocess(net, source.data[0])
        saveResult(net, i, end, vis)
        
    showResult(net, vis)
   # cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
   # cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
   # cv2.imshow("test",pimg)

    # returning the resulting image
    #return deprocess(net, source.data[0])
	####################################
# imports
import PIL.Image
import numpy as np
import scipy.ndimage as nd
import cv2
import caffe


# If your GPU supports CUDA and Caffe was built with CUDA support,
# call the following to run Caffe operations on the GPU.
def use_gpu():
    caffe.set_mode_gpu()
    caffe.set_device(0)  # select GPU device if multiple devices exist


def load_model():
    model_path = 'models/bvlc_googlenet/'  # substitute your path here
    net_fn = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    net = caffe.Classifier(net_fn, param_fn,
                           mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
                           channel_swap=(2, 1, 0))  # the reference model has channels in BGR order instead of RGB

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
    vis = vis * (255.0 / np.percentile(vis, 99.98))
    vis = np.uint8(np.clip(vis, 0, 255))

    pimg = PIL.Image.fromarray(vis)

    f = "frames/frame" + str(index) + end.replace('/', '-') + ".jpg"
    pimg.save(f, 'jpeg')
    print f


def showResult(vis):
    # adjust image contrast and clip

    vis = vis * (255.0 / np.percentile(vis, 99.98))
    vis = np.uint8(np.clip(vis, 0, 255))

    # pimg = PIL.Image.fromarray(vis)
    # pimg.show(),

    showOpenCV(vis)


def showOpenCV(image):
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#I think this is being converted both ways ...
    cv2.imshow("test", bgr)
    cv2.waitKey(0)  # Scripting languages are weird, It will not display without this
    cv2.destroyAllWindows()


def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


def objective_L2(dst):
    dst.diff[:] = dst.data


def make_step(net, end, sigma, step_size=1.5, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data']  # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += step_size / np.abs(g).mean() * g

    src.data[0] = blur(src.data[0], sigma)


def deepdream(net, base_img, iter_n=10, end='inception_4e/output', **step_params):
    # reshape and load image
    source = net.blobs['data']
    h, w, c = base_img.shape[:]
    source.reshape(1, 3, h, w)
    source.data[0] = preprocess(net, base_img)

    startSig = 1.0
    endSig = 0.1

    for i in xrange(iter_n):
        sigma = startSig + ((endSig - startSig) * i) / iter_n
        make_step(net, end=end, sigma=sigma, **step_params)

        # visualization
        vis = deprocess(net, source.data[0])
        saveResult(net, i, end, vis)

    showResult(vis)

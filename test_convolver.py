from unittest import TestCase

import numpy as np
import convolver
import setup_caffe_network as su


def setup_model():

    prototxt_path   = 'models/bvlc_googlenet/deploy.prototxt'
    caffemodel_path = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'  # this model comes with caffe
    pixel_mean = np.float32([104.0, 116.0, 122.0])  # ImageNet mean, training set dependent
    height = 224
    width = 224

    caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
    return caff.get_network()


layers = [
    {
        'layer':'inception_3a/output',
        'iter_n':5,
        'start_sigma':0.3,
        'end_sigma':0.0,
        'start_step_size':6,
        'end_step_size':3
    },

]


# This is a functional test, it runs pretty fast on a GPU and tests most of convolver.py
class TestConvolver(TestCase):
    def test_deepmod(self):
        su.SetupCaffe.gpu_on()
        net = setup_model()
        co = convolver.Convolver(net)
        vis = co.deepmod('ImagesIn/soft-grey.jpg', layers)
        assert vis.shape == (94, 94, 3)
        assert np.array_equal(vis[10, 10], [179, 143, 165])
        assert np.array_equal(vis[43, 34], [208, 171, 193])
        assert np.array_equal(vis[76, 89], [188, 193, 194])

        test_vis = np.load('test/test_vis.npy')
        assert np.array_equal(vis, test_vis)


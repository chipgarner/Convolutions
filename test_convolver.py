from unittest import TestCase

import numpy as np
import convolver
import setup_caffe_network as su
import models as ml


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
        net = ml.NetModels.setup_googlenet_model('models/')
        co = convolver.Convolver(net)
        vis = co.deepmod('ImagesIn/soft-grey.jpg', layers)
        assert vis.shape == (94, 94, 3)
        assert np.array_equal(vis[10, 10], [179, 143, 165])
        assert np.array_equal(vis[43, 34], [208, 171, 193])
        assert np.array_equal(vis[76, 89], [188, 193, 194])

        test_vis = np.load('test/test_vis.npy')
        assert np.allclose(vis, test_vis, 0.01)


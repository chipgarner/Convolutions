import numpy as np
import convolver
import setup_caffe_network as su
import models as ml


layers = [
    {
        'layer': 'inception_4c/output',
        'iter_n': 5,
        'start_sigma': 0.9,
        'end_sigma': 0.2,
        'start_step_size': 6,
        'end_step_size': 1
    },

]


su.SetupCaffe.gpu_on()
co = convolver.Convolver(ml.NetModels.setup_places_model(''))
co.deepmod('ImagesIn/youandme.jpg', layers)
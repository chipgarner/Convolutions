import convolver
import setup_caffe_network as su
import models as ml


layers = [
    {
        'layer': 'inception_3a/output',
        'iter_n': 5,
        'start_sigma': 0.3,
        'end_sigma': 0.0,
        'start_step_size': 6,
        'end_step_size': 3
    },

]


su.SetupCaffe.gpu_on()
co = convolver.Convolver(ml.NetModels.setup_googlenet_model(''))
co.deepmod('ImagesIn/1920x1080.jpg', layers)

import convolver
import sys
sys.path.insert(0, '/dev/projects/CommonCaffe')
import models as ml


layers = [
    {
        'layer': 'inception_5b/output',
        'iter_n': 20,
        'start_sigma': 0.3,
        'end_sigma': 0.0,
        'start_step_size': 6,
        'end_step_size': 3
    },

]


co = convolver.Convolver(ml.NetModels.setup_googlenet_model('models/'))
co.deepmod('ImagesIn/vg.bmp', layers)

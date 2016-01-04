import convolver
import models as ml


co = convolver.Convolver(ml.NetModels.setup_vgg('../CommonCaffe/TrainedModels/'))


layers = [
    {
        'layer':'conv5_4',
        'iter_n': 30,
        'start_sigma':0.3,
        'end_sigma':0.0,
        'start_step_size':10,
        'end_step_size':10
    },

]


co.deepmod('ImagesIn/soft-grey.jpg', layers)
import convolver
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


co = convolver.Convolver(ml.NetModels.setup_places_model('../CommonCaffe/TrainedModels/'))
co.deepmod('ImagesIn/youandme.jpg', layers)
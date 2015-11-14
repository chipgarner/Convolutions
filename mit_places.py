import numpy as np
import convolver
import setup_caffe_network as su


def setup_model():
    # From the MIT Places paper: B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva.
    # Learning Deep Features for Scene Recognition using Places Database.
    # Advances in Neural Information Processing Systems 27 (NIPS), 2014
    # AKA the "NIPS 2014 Paper".  GoogleNet trained on places images.
    prototxtPath   = 'models/googlenet_places205/deploy_places205.protxt'
    caffemodelPath = 'models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel'  # From the model zoo
    pixel_mean = np.float32([104.0, 116.0, 122.0]) # WRONG training set! ImageNet mean, training set dependent
    height = 224
    width = 224

    caff = su.SetupCaffe(prototxtPath, caffemodelPath, pixel_mean, height, width)
    return caff.get_network()

su.SetupCaffe.gpu_on()
co = convolver.Convolver(setup_model())


layers = [
    {
        'layer':'inception_4c/output',
        'iter_n': 500,
        'start_sigma': 0.9,
        'end_sigma': 0.2,
        'start_step_size': 6,
        'end_step_size': 1
    },

]


co.deepmod('ImagesIn/youandme.jpg', layers)
import convolver as co

co.use_gpu()
net = co.load_model()
img = co.load_image('ImagesIn/dianaFits.jpg')

ending = 'inception_5b/output'
co.deepdream(net, img, iter_n = 30, end = ending)
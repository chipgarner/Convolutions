import convolver as co

co.use_gpu()
net = co.load_model()
img = co.load_image('dianaFits.jpg')

ending = 'inception_5a/output'
co.deepdream(net, img, iter_n = 5, end = ending)
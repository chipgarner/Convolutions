import convolver as co

co.use_gpu()
net = co.load_model()
img = co.load_image('youandme.jpg')

ending = 'inception_4e/output'
co.deepdream(net, img, iter_n = 30, end = ending)
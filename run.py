import convolver as co

co.use_gpu()
net = co.load_model()
img = co.load_image('youandme.jpg')
co.deepdream(net, img)
#This overwrites everything in the frames directory, rename it if you want to save the images.
import convolver as co

co.use_gpu()
net = co.load_model()
img = co.load_image('dianaFits.jpg')

ending = 'conv2/3x3_reduce'
co.deepdream(net, img, iter_n = 30, end = ending)

ending = 'conv2/3x3'
co.deepdream(net, img, iter_n = 30, end = ending)

ending = 'conv2/norm2'
co.deepdream(net, img, iter_n = 30, end = ending)


for i in range(3, 6):
	ending = 'inception_' + str(i) + 'a/1x1'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'a/3x3_reduce'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'a/3x3'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'a/5x5_reduce'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'a/5x5'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'a/output'
	co.deepdream(net, img, iter_n = 30, end = ending)
	
	ending = 'inception_' + str(i) + 'b/1x1'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'b/3x3_reduce'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'b/3x3'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'b/5x5_reduce'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'b/5x5'
	co.deepdream(net, img, iter_n = 30, end = ending)
	ending = 'inception_' + str(i) + 'b/output'
	co.deepdream(net, img, iter_n = 30, end = ending)


ending = 'inception_4c/1x1'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4c/3x3_reduce'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4c/3x3'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4c/5x5_reduce'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4c/5x5'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4c/output'
co.deepdream(net, img, iter_n = 30, end = ending)

ending = 'inception_4d/1x1'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4d/3x3_reduce'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4d/3x3'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4d/5x5_reduce'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4d/5x5'
co.deepdream(net, img, iter_n = 30, end = ending)
ending = 'inception_4d/output'
co.deepdream(net, img, iter_n = 30, end = ending)


#this layer takes more GPU memory
#ending = 'inception_4e/1x1'
#co.deepdream(net, img, iter_n = 30, end = ending)
#ending = 'inception_4e/3x3_reduce'
#co.deepdream(net, img, iter_n = 30, end = ending)
#ending = 'inception_4e/3x3'
#co.deepdream(net, img, iter_n = 30, end = ending)
#ending = 'inception_4e/5x5_reduce'
#co.deepdream(net, img, iter_n = 30, end = ending)
#ending = 'inception_4e/5x5'
#co.deepdream(net, img, iter_n = 30, end = ending)
#ending = 'inception_4e/output'
#co.deepdream(net, img, iter_n = 30, end = ending)
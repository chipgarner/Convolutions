# imports
import numpy as np
import scipy.ndimage as nd
import sys
sys.path.insert(0, '/dev/projects/CommonCaffe')
import display
import cv2


class Convolver:
    def __init__(self, net):
        self.net = net

    # reshape and load image
    def __load_image(self, image_relative_path):
        img = cv2.imread(image_relative_path)  # np.float32(PIL.Image.open(image_relative_path))
        self.source = self.net.blobs['data']
        h, w, c = img.shape[:]
        print str(c)
        self.source.reshape(1, 3, h, w)
        self.source.data[0] = self.__preprocess(img)
        return img

    # a couple of utility functions for converting to and from Caffe's input image layout
    def __preprocess(self, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - self.net.transformer.mean['data']

    def __deprocess(self, img):
        return np.dstack((img + self.net.transformer.mean['data'])[::-1])

    def __save_result(self, index, image_path, end, vis):
        # pimg = PIL.Image.fromarray(vis)

        # get the image name from the path
        txt = image_path.split('/')
        nm = txt[len(txt) - 1].split('.')
        name = nm[0]

        f = "frames/" + str(index) + '_' + name + '_' + end.replace('/', '-') + ".jpg"
        # pimg.save(f, 'jpeg')
        print f

    def __blur(self, img, sigma):
        # vis = self.__deprocess(img)
        # cv2.
        # vis = cv2.bilateralFilter(vis,  40, 20, 6) # Really slow!
        # img = self.__preprocess(vis)
        if sigma > 0:
            img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
            img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
            img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
        return img

    def __objective_L2(self, dst):
        dst.diff[:] = dst.data

    def __make_step(self, end, sigma, step_size=1.5):
        '''Basic gradient ascent step.'''

        src = self.net.blobs['data']  # input image is stored in Net's 'data' blob
        dst = self.net.blobs[end]

        self.net.forward(end=end)
        self.__objective_L2(dst)  # specify the optimization objective
        self.net.backward(start=end)
        g = src.diff[0]

        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = self.__blur(src.data[0], sigma)

    def __visualize(self):
        vis = self.__deprocess(self.source.data[0])
        # vis = cv2.bilateralFilter(vis,  20, 50, 4)
        # adjust image contrast and clip
        vis = vis * (255.0 / np.percentile(vis, 99.98))
        vis = np.uint8(np.clip(vis, 0, 255))
        return vis

    def deepmod(self, image_path, layers):

        self.__load_image(image_path)

        for e, o in enumerate(layers):
            end_layer = o['layer']

            for i in xrange(o['iter_n']):
                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                self.__make_step(end=end_layer, sigma=sigma, step_size=step_size)

                vis = self.__visualize()
                self.__save_result(i, image_path, end_layer, vis)
                # vis = cv2.bilateralFilter(vis,  20, 50, 4)
                # Above works, feed it back into the network
            display.Display().showResultCV(vis)
            # display.Display().showResultCV(vis)
            return vis

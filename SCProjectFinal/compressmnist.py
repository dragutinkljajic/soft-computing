import numpy as np

from skimage.measure import label
from skimage.measure import regionprops

from skimage.io import imread
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from scipy.fftpack import dct, idct

class Dummy(object): pass


def reshapeMnistNumber(img):
    img = img.reshape(28, 28)/255.0
    th = img > 0

    labeled_img = label(th)
    regions = regionprops(labeled_img)

    top = min([region.bbox[0] for region in regions])
    left = min([region.bbox[1] for region in regions])

    reshaped = np.zeros((20, 20))
    if top < 8 and left < 8:
        reshaped[0:20, 0:20] = img[top:top+20, left:left+20]
    elif top < 8:
        reshaped[0:20, 0:28-left] = img[top:top+20, left:28]
    elif left < 8:
        reshaped[0:28-top, 0:20] = img[top:28, left:left+20]
    else:
        reshaped[0:28-top, 0:28-left] = img[top:28, left:28]

    return reshaped


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')

    for idx, res in enumerate(labels):
        retVal[idx, res] = 1
    return retVal


def compressMnist(mnist):
    compress = Dummy()
    compress.data = []
    compress.target = []

    for idx, pic in enumerate(mnist.data):
        img = reshapeMnistNumber(pic).reshape(400)
        imgdata = np.squeeze(np.asarray(img))
        compress.data.append(imgdata)

        compress.target.append(to_categorical(mnist.target[idx:idx+1], 10).reshape(10))

    compress.data = np.array(compress.data)
    compress.target = np.array(compress.target)

    return compress

from skimage.io import imread
from scipy import ndimage
import cv2
import numpy as np
from skimage.color import rgb2gray

from skimage.measure import label
from skimage.measure import regionprops

#from vector import distance, pnt2line

from compressmnist import *
from neuralnetwork import *
from analyze import *

mnist = fetch_mldata('MNIST original')
compress = compressMnist(mnist)

#network = initializeNetwork(compress)

network = load('network.obj')

#test(network, compress)

#for i in range(0, 9):
    #line = detect_line('Genericki projekat - level 1/video-' + str(i) + '.avi')
#    line = detect_line_hough('Genericki projekat - level 1/video-' + str(i) + '.avi')

#    counter = detect_numbers('Genericki projekat - level 1/video-' + str(i) + '.avi', line, network)

#brisanje sadrzaja u outu pre svakog pokretanja
open('out.txt', 'w').close()

with open('res.txt', 'r') as res, open('out.txt', 'w') as out:
    data = res.read()
    lines = data.split('\n')

    out.write('RA79/2013 Dragutin Kljajic\n')
    out.write('file\tsum\n')

    for id, lines in enumerate(lines):
        if(id>0):
            cols = lines.split('\t')

            if(cols[0] != ''):
                #line = detect_line_hough('Genericki projekat - level 1/' + cols[0])
                #counter = detect_numbers('Genericki projekat - level 1/' + cols[0], line, network)

                line = detect_line_hough('Genericki projekat - level 2/' + cols[0])
                counter = detect_numbers('Genericki projekat - level 2/' + cols[0], line, network)
                out.write(cols[0] + '\t' + str(counter) + '\n')


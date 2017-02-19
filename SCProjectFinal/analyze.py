from skimage.io import imread
from scipy import ndimage
import cv2
import numpy as np
from skimage.color import rgb2gray

from skimage.measure import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt

from vector import distance, pnt2line
from neuralnetwork import *

cc = -1
def nextId():
    global cc
    cc += 1
    return cc


def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

# -----------------------------INITIALIZE--------------------
# color filter
kernel = np.ones((2, 2), np.uint8)
# boundary for color filter
boundaries = [
    ([230, 230, 230], [255, 255, 255])
]
# ------------------------------------------------------------


def reshapeNumber(img):
    segmented = img>0

    labeled_img = label(segmented)
    regions = regionprops(labeled_img)

    top = min([region.bbox[0] for region in regions])
    left = min([region.bbox[1] for region in regions])

    retVal = np.zeros((20, 20))
    if top < 8 and left < 8:
        retVal[0:20, 0:20] = img[top:top+20, left:left+20]
    elif left < 8:
        retVal[0:(28-top), 0:20] = img[top:28, left:left + 20]
    elif top < 8:
        retVal[0:20, 0:(28-left)] = img[top:top+20, left:28]
    else:
        retVal[0:(28 - top), 0:(28-left) ] = img[top:28, left:28]

    return retVal


def detect_line(path):
    cap = cv2.VideoCapture(path)

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res_line = cv2.bitwise_and(frame, frame, mask=mask)
    gray_line = rgb2gray(res_line)
    th_line = gray_line > 0
    img_tmp = th_line * 1.0

    img_tmp = cv2.dilate(img_tmp, kernel)
    img_tmp = cv2.dilate(img_tmp, kernel)
    img_tmp = cv2.erode(img_tmp, kernel)
    img_tmp = cv2.erode(img_tmp, kernel)

    regions_line = regionprops(label(img_tmp))

    for region in regions_line:
        line = [(region.bbox[1], region.bbox[2]), (region.bbox[3], region.bbox[0])]

    cap.release()

    print "line bezobrazno: ", line

    return line


def detect_line_hough(path):
    cap = cv2.VideoCapture(path)

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, 600, 10)

    lowerx = 0
    lowery = 0
    uppery = 0
    upperx = 0

    for x1, y1, x2, y2 in lines[0]:
        lowerx = x1
        lowery = y1
        upperx = x2
        uppery = y2

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if x1 < lowerx:
                lowerx = x1
                lowery = y1
            if x2 > upperx:
                uppery = y2
                upperx = x2

    line = [(lowerx, lowery), (upperx, uppery)]
    cap.release()

    return line


def detect_numbers(path, line, network):
    elements = []
    counter = 0

    cap = cv2.VideoCapture(path)

    while True:
        ret, img = cap.read()

        if ret is False:
            break

        # -------------- PREPOZNAVANJE BROJEVA ---------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of white color in HSV
        # change it according to your need !
        sensitivity = 40
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.dilate(res, kernel)

        res_gray = rgb2gray(res)

        # -----------------------------------------------------------------

        # ------------------- DETEKCIJA REGIONA ---------------------------

        (lower, upper) = boundaries[0]

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img_reg = 1.0 * mask

        img_reg = cv2.dilate(img_reg, kernel)
        img_reg = cv2.dilate(img_reg, kernel)

        labeled, nr_objects = ndimage.label(img_reg)
        objects = ndimage.find_objects(labeled)
        # ----------------------------------------------------------------
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if dxc > 11 or dyc > 11:
                cv2.rectangle(img, (xc - 14, yc - 14), (xc + 14, yc + 14), (25, 25, 255), 1)
                retVal = np.zeros((28, 28))

                elem = {'center': (xc, yc), 'size': (dxc, dyc), 'value': -1}
                # find in range
                lst = inRange(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['dictionary'] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
                    if (xc + 14 < 640) and (yc + 14 < 480) and (xc - 14 > 0) and (yc - 14 > 0):
                        retVal[0:28, 0:28] = res_gray[yc - 14:yc + 14, xc - 14:xc + 14]

                        retVal = reshapeNumber(retVal)

                        retVal = retVal.reshape(400)
                        retVal = np.squeeze(np.asarray(retVal))

                        pics = []
                        pics.append(retVal)
                        pics = np.array(pics)
                        numbers = predict(network, pics)

                        for num in numbers:
                            elem['dictionary'][num] += 1
                            elem['value'] = num

                    elem['id'] = nextId()
                    elem['pass'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc)}]
                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc)})

                    if (xc + 14 < 640) and (yc + 14 < 480) and (xc - 14 > 0) and (yc - 14 > 0):
                        retVal[0:28, 0:28] = res_gray[yc - 14:yc + 14, xc - 14:xc + 14]

                        retVal = reshapeNumber(retVal)

                        retVal = retVal.reshape(400)
                        retVal = np.squeeze(np.asarray(retVal))

                        pics = []
                        pics.append(retVal)
                        pics = np.array(pics)
                        numbers = predict(network, pics)

                        for num in numbers:
                            lst[0]['dictionary'][num] += 1
                            if lst[0]['value'] == -1:
                                lst[0]['value'] = num
                            if lst[0]['value'] != max(lst[0]['dictionary'], key=lambda j: lst[0]['dictionary'][j]):
                                lst[0]['value'] = max(lst[0]['dictionary'], key=lambda j: lst[0]['dictionary'][j])
        tmp_cnt = counter
        for el in elements:
            dist, pnt, r = pnt2line(el['center'], line[0], line[1])
            if r > 0:
                if dist < 5:
                    if el['pass'] is False:
                        el['pass'] = True
                        tmp_cnt += el['value']

        if tmp_cnt != counter:
            counter = tmp_cnt
            print 'Total sum: ', counter

        cv2.putText(img, 'Total sum: ' + str(counter), (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return counter

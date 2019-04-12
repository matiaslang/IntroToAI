# -*- coding: utf-8 -*-

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap
# Luetaan argumentit luokittelijalle ja testikuvalle
parser = ap.ArgumentParser()
parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())
# Ladataan luokittelija
clf, pp = joblib.load(args["classiferPath"])
# Käsitellään testikuvaa
im = cv2.imread(args["image"])
im = cv2.resize(im,(960,540))
# Muunnetaan mustavalkoiseksi kuvaksi ja toteutetaan Gaussinen suodatus
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
# Binarisoidaan kuva
ret, im_th = cv2.threshold(im_gray, 95, 255, cv2.THRESH_BINARY_INV) # (Säätääksesi mustavalkoistamisen raja-arvo, muuta tämän funktion toista argumenttia (oletusarvo 95))
# Etsitään kuvasta kiinnostavat alueet
image,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Sovitetaan suorakulmio kiinnostavien alueiden ympärille
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# Jokaiselle suorakulmiolle lasketaan HOG piirteet ja ennustetaan 
# luokittelijan perusteella mikä numero on kyseessä
for rect in rects:
    # Piirretään suorakulmiot numeroiden ympärille
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Muutetaan kuvan kokoa
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Lasketaan HOG piirteet
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2-Hys", visualize=False)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
    nbr = clf.predict(roi_hog_fd)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
cv2.imshow("The output image", im)
cv2.imshow("The thresholded image", im_th)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('output-{}-{}.jpg'.format(args["image"][:-4],args["classiferPath"][:-4]),im)

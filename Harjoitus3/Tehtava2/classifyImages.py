import cv2
import random
import numpy as np
import argparse as ap
from sklearn.externals import joblib
from skimage.feature import hog
import glob
from imutils import build_montages

def callback(x):
    pass

def log_function(values):
    return np.log(np.abs(values))*np.sign(values)

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
    args = vars(parser.parse_args())
    # Ladataan luokittelija
    clf = joblib.load(args["classiferPath"])
    filelist = glob.glob('output_test_images/*')
    images = []
    for filename in filelist:
        # Muutetaan kuvan kokoa
        im = cv2.imread(filename)
        im = cv2.resize(im,(320,480))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Lasketaan Hu moment invariantit
        humoments = log_function(cv2.HuMoments(cv2.moments(im_gray)).flatten())
        nbr = clf.predict([humoments])
        cv2.putText(im, str(nbr[0]), (160,440),cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)
        images.append(im)
    num_im_row = round((len(filelist))**(0.5) + 0.5)
    num_im_col = round(len(filelist)/num_im_row + 0.5)
    montages = build_montages(images, (int(640/num_im_row), int(960/num_im_col)), (num_im_row, num_im_col))
    for montage in montages:
        cv2.imshow("Montage", montage)
        cv2.waitKey(0)
    cv2.imwrite('output_{}.jpg'.format(args["classiferPath"][:-4]),montage)
    




import glob
import pickle
import numpy as np
from PIL import Image
import shutil
import os

def overwrite_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir)

if __name__=='__main__':
    overwrite_dir('output_test_images')
    overwrite_dir('output_train_images')
    # Määritetään testijoukon koko ja sekoitetaan alkuperäinen data
    testsetsize = 0.1
    filelist = glob.glob('output_images/*.bmp')
    if not filelist:
        print('output_images is empty! Run first files testTransformation.py and runTransformation.py')
        exit()
    numtestimages = int(testsetsize * len(filelist) + 0.5)
    labels = []
    for fname in filelist:
        labels.append(fname[21])
    y=np.array(labels)
    images = []
    for fname in filelist:
        images.append(np.array(Image.open(fname)))
    x=np.array(images, "int16")
    indeces = np.arange(y.shape[0])
    np.random.shuffle(indeces)
    x = x[indeces]
    y = y[indeces]
    filelist = np.asarray(filelist)
    filelist = filelist[indeces]
    filelist = filelist.tolist()

    # Luodaan testijoukko
    test_indeces = indeces[0:numtestimages]
    x_test = x[test_indeces]
    y_test = y[test_indeces]
    test_filenames = np.asarray(filelist)
    test_filenames = test_filenames[test_indeces]
    test_filenames = test_filenames.tolist()
    set_ = [x_test, y_test]
    pickle.dump(set_, open('testdataset.pkl', 'wb'), protocol=2)
    for filename in test_filenames:
        shutil.copy(filename, "output_test_images/{}".format(filename[13:]))
        
    # Luodaan opetusjoukko
    train_indeces = indeces[numtestimages:]
    x_train = x[train_indeces]
    y_train = y[train_indeces]
    train_filenames = np.asarray(filelist)
    train_filenames = train_filenames[train_indeces]
    train_filenames = train_filenames.tolist()
    set_ = [x_train, y_train]
    pickle.dump(set_, open('traindataset.pkl', 'wb'), protocol=2)
    for filename in train_filenames:
        shutil.copy(filename, "output_train_images/{}".format(filename[13:]))
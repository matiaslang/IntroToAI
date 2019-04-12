import cv2
import numpy as np
import glob
import pickle
import os
import shutil

def callback(x):
    pass

filelist = glob.glob('input_images/*.jpg')
filelist.sort()
labels = []
for filename in filelist:
    labels.append(filename[13:21])
classes = list(set(labels))
classes.sort()
counter = [0] * len(classes)
for filename in filelist:
    classname = filename[13:21]
    index = classes.index(classname)
    counter[index] += 1
    # Muutetaan kuvan kokoa
    im = cv2.imread(filename) 
    im = cv2.resize(im,(320,480))
    # Otetaan ylös hsvvalues.txt tiedostosta säädetyt hsv-värimallin arvot
    try:
        with open ('hsvvalues.txt', 'rb') as fp:
            [MinH, MaxH, MinS, MaxS, MinV, MaxV] = pickle.load(fp)
    except:
        print("You should probably run the file testTransformation.py first!")
        shutil.rmtree('output_images/*', ignore_errors=True)
        break
    # Sumennetaan kuvaa
    blur = cv2.blur(im,(3,3))
    # Toteutetaan bgr-hsv muunnos ja AND operaatio
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([0, MinS, MinV])
    upper = np.array([MaxH, MaxS, MaxV])
    mask2 = cv2.inRange(hsv,lower,upper)
    lower1 = np.array([MinH, MinS, MinV])
    upper1 = np.array([255, MaxS, MaxV])
    mask1 = cv2.inRange(hsv,lower1,upper1)
    mask = cv2.bitwise_or(mask1,mask2)   
    # Toteutetaan morfologisia operaatioita suodattaakseen pois taustakohina     
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation3,5)
    ret,thresh = cv2.threshold(median,120,255,0)
    im2, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area=100
    ci=0	
    # Etsitään mustavalkoistetun kuvan suurin alue
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i            		  
    cnts = contours[ci]
    rect = cv2.boundingRect(cnts)
    x1,y1,w1,h1 = rect
    cv2.rectangle(im, (x1,y1),(x1+w1,y1+h1), (0,255,0), 3)
    # Selvitään kämmenen keskipiste
    maxdistance=0
    pt=(0,0)
    for index_y in range(int(y1+0.1*h1),int(y1+0.8*h1)):
        for index_x in range(int(x1+0.1*w1),int(x1+0.8*w1)):
            distance=cv2.pointPolygonTest(cnts,(index_x,index_y), True)
            if(distance>maxdistance):
                maxdistance=distance
                pt = (index_x,index_y)

    radius = int(maxdistance)
    cv2.circle(im,pt,radius,(255,0,0),2)
    cv2.rectangle(im, (x1,y1),(x1+w1,pt[1]+radius), (0,0,255), 3) 
    cropped_image = thresh[y1:pt[1]+radius,x1:x1+w1]
    # Skaalataan muunnetun kuvan kokoa
    cropped_image = cv2.resize(cropped_image, (100, 150))
    outputfilename = "output_images/gesture{}_{}.bmp".format(index+1, counter[index])
    cv2.imwrite(outputfilename, cropped_image)
cv2.destroyAllWindows()


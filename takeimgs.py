# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:32:34 2022

@author: lenovo
"""
import os
#import glob
import numpy as np
import cv2 as cv
#The fomate of the taken images should be set to 480*640

load = np.load(os.path.join('D:/Bristol Robotics/Dissertation/img', "mtx_dist.npz"))# load the undistortion model 
mtx = load["mtx"]
print(mtx)
dist = load["dist"]
print(dist)

cap = cv.VideoCapture(0)
i=0
while(1):
    _, img = cap.read()
    
    dst = cv.undistort(img, mtx, dist, None)
    scale_percent = 224/480 # adjust thsi when you need different size of input
    width = int(dst.shape[1] * scale_percent )
    height = int(dst.shape[0] * scale_percent)
    dim = (width, height)
     
    # resize image
    resized = cv.resize(dst, dim, interpolation = cv.INTER_AREA)
    #cropImg = resized[0:1024,0:1024]
    if i==1:
        break
    else:
        cv.imwrite('D:\Bristol Robotics\Dissertation\COMP341-A1-master\Data\grasping'+str(i)+'_RGB.png',resized)
        i+=1
    cv.imshow("capture", resized)
cap.release()
cv.destroyAllWindows()    


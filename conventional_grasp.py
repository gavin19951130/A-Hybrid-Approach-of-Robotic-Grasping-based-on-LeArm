# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 18:32:43 2022

@author: lenovo
"""


from keras.models import load_model
import os
import numpy as np
import cv2 as cv
import math
import serial
import time
import pandas as pd
import sys

############################The trnsformation from Servo Potential to joint angles###########################
def angle(x):
  y = (x - 500) * np.pi/2000
  return y

def anglej1(x):
  y = (x - 1500) * np.pi/1000
  return y

def arcangle(y):
  x = (y *2000 / np.pi) + 500
  return x


############################Measured robot arm lengths########################################################
l1 = 2.8     #cm
l2 = 10.4
l3 = 9.6
l4 = 0.0
l5 = 16.3




############################The joint angles when the gripper is placed at its initial position########################################################
q1 = angle(1454)    #ID6 
q2 = angle(1686)    #ID5
q3 = angle(630)     #ID4
q4 = angle(2059)    #ID3
q5 = angle(1453)    #ID2 


############################The trasformation from the robot coordinate frame to the kinematics coordinate frame############################
qfk1 = q1 - np.pi/2
qfk2 = np.pi - q2 
qfk3 = q3 - np.pi/2
qfk4 = np.pi - q4
qfk5 = q5
fi = qfk2 + qfk3 + qfk4

############################The offset to calibrate the camera reading########################################################

gripper_x = -5.289237648785177 + 0.04049203835173287#(centimeter) Please set the value yourself
gripper_y = 15.56917479889851 + 0.04382551971693616#(centimeter)
gripper_t = 1.575751480016183 + 0.025609734076050605

############################The inverse kinematics(The closest pose to the robot's initial pose)########################################################
def IK2(x,y,z):
    
    theta1 = np.arctan2(y,x)
    X = x - l5*np.sin(fi)*np.cos(theta1)
    Y = y - l5*np.sin(fi)*np.sin(theta1)
    Z = z + l5*np.cos(fi) -l1
    c3 = (X**2 + Y**2 + Z**2 - l2**2 - l3**2)/(2*l2*l3)
    theta3 = np.arctan2(math.sqrt(1-c3**2),c3)
    b = np.arctan2(Z, math.sqrt(X**2 + Y**2))
    a = np.arctan2(l3 * np.sin(theta3), (l2+l3*np.cos(theta3)))
    theta2 = b-a
    theta4 = fi -theta2 - theta3
    theta3 = np.arctan2(math.sqrt(1-c3**2), c3)
    
    theta7 = theta1
    theta6 = theta2
    theta5 = theta3
    theta4 = theta4
    
    return theta7, theta6, theta5, theta4
############################The forward kinematics########################################################

def FK(qfk1,qfk2,qfk3,qfk4,qfk5):
    global  fk_T
    
###### The DH table##########
    a = np.array([0,l2,l3,0,0])
    ap = np.array([90.0*np.pi/180.0,0.0,0.0,90.0*np.pi/180.0,0.0])
    d = np.array([l1,0,0,0,l5])

    c1 = np.cos(qfk1)
    c2 = np.cos(qfk2)
    c3 = np.cos(qfk3)
    c4 = np.cos(qfk4)
    c5 = np.cos(qfk5)
    
    s1 = np.sin(qfk1)
    s2 = np.sin(qfk2)
    s3 = np.sin(qfk3)
    s4 = np.sin(qfk4)
    s5 = np.sin(qfk5)

    apc1 = np.cos(ap[0])
    apc2 = np.cos(ap[1])
    apc3 = np.cos(ap[2])
    apc4 = np.cos(ap[3])
    apc5 = np.cos(ap[4])

    aps1 = np.sin(ap[0])
    aps2 = np.sin(ap[1])
    aps3 = np.sin(ap[2])
    aps4 = np.sin(ap[3])
    aps5 = np.sin(ap[4])

    T01 = np.array([
          [c1,-apc1*s1,aps1*s1,a[0]*c1],
          [s1,apc1*c1,-aps1*c1,a[0]*s1],
          [0,aps1,apc1,d[0]],
          [0,0,0,1]
      ])
    #print('T01:',T01)
    
    T12 = np.array([
          [c2,-apc2*s2,aps2*s2,a[1]*c2],
          [s2,apc2*c2,-aps2*c2,a[1]*s2],
          [0,aps2,apc2,d[1]],
          [0,0,0,1]
      ])
    #print('T12:',T12)
    #print('T02:',T01 * T12)
    T23 = np.array([         
          [c3,-apc3*s3,aps3*s3,a[2]*c3],
          [s3,apc3*c3,-aps3*c3,a[2]*s3],
          [0,aps3,apc3,d[2]],
          [0,0,0,1]
      ])
    
    
    T34 = np.array([         
          [c4,-apc4*s4,aps4*s4,a[3]*c4],
          [s4,apc4*c4,-aps4*c4,a[3]*s4],
          [0,aps4,apc4,d[3]],
          [0,0,0,1]
      ])

    T45 = np.array([         
          [c5,-apc5*s5,aps5*s5,a[4]*c5],
          [s5,apc5*c5,-aps5*c5,a[4]*s5],
          [0,aps5,apc5,d[4]],
          [0,0,0,1]
      ])
    
    T02 = np.dot(T01, T12)
    T03 = np.dot(T02, T23)
    T04 = np.dot(T03, T34)
    T05 = np.dot(T04, T45)
    fk_T = T05
    
    R_base2gripper = fk_T[0:3,:3]
    t_base2gripper = fk_T[0:3,3]

    return R_base2gripper, t_base2gripper

############################Send the control parameters to Arduino########################################################
def move(theta6, theta5, theta4, theta3):
    ser = serial.Serial('COM3',9600,timeout=1) #Please set the baud rate and the port name yourself
    
    starter0 = 'a'
    starter1 = 'b'
    starter2 = 'c'
    starter3 = 'e'

    
    ender0 = 'w'
    ender1 = 'x'
    ender2 = 'y'
    ender3 = 'z'

    
    theta6 = starter0 + str(round(theta6,3)) + ender0
    theta5 = starter1 + str(round(theta5,3)) + ender1
    theta4 = starter2 + str(round(theta4,3)) + ender2
    theta3 = starter3 + str(round(theta3,3)) + ender3

    
    theta = theta6 + theta5 + theta4 + theta3
    
    i=0

    while i<=6: #6(seconds): The time scale to reveice the control parameters, adjust this according to the number of parameters and ensure all of them are sent
    
        _ = ser.write(theta.encode('utf-8'))
        time.sleep(1) 
        val = ser.readline().decode('utf-8')
        print(val)# The print lines of Arduino will be printed in the console here
        i+=1
        
############################The detection of the ArUco marker############################
def arucodetect(dst, gripper_x, gripper_y, mtx, dist):  
     num = sum_x = sum_y = 0
     global cam2aruco_x, cam2aruco_y, cam2aruco_t
     gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    # Detect aruco blocks
     corners, ids, rejectImaPoint = cv.aruco.detectMarkers(
        gray, cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250), parameters = cv.aruco.DetectorParameters_create() # Initialize the detector parameters
    )
     cv.aruco.drawDetectedMarkers(gray, corners, ids)
     if len(corners) > 0:
        if ids is not None:
            # Get information from aruco
            cv.aruco.drawDetectedMarkers(gray, corners, ids)
            ret = cv.aruco.estimatePoseSingleMarkers(
                corners, 0.00022, mtx, dist
            )# 0.0022 is marker length (meter)
            # Get the rotation vector and the translation vector
            (rvec, tvec) = (ret[0], ret[1])
            (rvec - tvec).any()
            xyz = tvec[0, 0, :]
            """
               # calculate the coordinates of the aruco relative to the gripper
                 gripper_y, gripper_x represent the total offset of the camera relative to the gripper and the error occurred during identification
                 The grabbing effect can be corrected by modifying the offsets of gripper_y and gripper_x;
            """
            xyz = [xyz[0]*10000+gripper_y, xyz[1]*10000+gripper_x, xyz[2]*10000]
            
            R=np.zeros((3,3),dtype=np.float64)
            cv.Rodrigues(rvec,R)
            sy=math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
            singular=sy< 1e-6
            if not singular:
                px = math.atan2(R[2, 1], R[2, 2]) 
                py = math.atan2(-R[2, 0], sy)
                pz = math.atan2(R[1, 0], R[0, 0])
            else:
                px = math.atan2(-R[1, 2], R[1, 1])
                py = math.atan2(-R[2, 0], sy)
                pz = 0
                
            for i in range(rvec.shape[0]):
                # draw aruco in the picture
                cv.aruco.drawDetectedMarkers(dst, corners)
                result_img = cv.drawFrameAxes(
                    dst,
                    mtx,
                    dist,
                    rvec[i, :, :],
                    -tvec[i, :, :],
                    0.0003, # the length of the coordiante axis drawn
                )
                
                sum_x += xyz[1]
                sum_y += xyz[0]
                num += 1
                
                x = xyz[1]
                y = xyz[0]
                pz = pz + gripper_t
                print("x,y,pz: ",x,y,pz)
                cam2aruco_x, cam2aruco_y, cam2aruco_t = x, y, pz
                
            cv.imshow("encode_image", result_img)
            
############main function#################################################################
model = load_model('D:/Bristol Robotics/Dissertation/master/Data/MLPtesting/dl.h5') # load the trained MLP model 
load = np.load(os.path.join('D:/Bristol Robotics/Dissertation/img', "mtx_dist.npz")) # The path of the trained model
mtx = load["mtx"]
print(mtx)
dist = load["dist"]
print(dist)
datax = []
datay = []
datat = []
theta6list = []
theta5list = []
theta4list = []
theta3list = []

gripper2cam_x, gripper2cam_y = 0,0 #the initial position of gripper

############Get the accurate transformation matrix from robot base to the gripper's initial position########################
R_base2gripper, t_base2gripper = FK(qfk1,qfk2,qfk3,qfk4,qfk5) #qfk1,qfk2,qfk3,qfk4,qfk5 are the angles of each joint when the gripper is in its initial position
base2aruco_fi = fi
base2gripper_x, base2gripper_y,  base2gripper_z = t_base2gripper[0] , t_base2gripper[1] , t_base2gripper[2]

cap = cv.VideoCapture(0)
while cv.waitKey(1) != ord("q"):
    _, img = cap.read()
    h, w = img.shape[:2]
    dst = cv.undistort(img, mtx, dist, None)          
    cv.imshow("undistorted _image", dst)
    arucodetect(dst, gripper_x, gripper_y, mtx, dist)
    
    
############Please comment on the below lines of code first and calibrate the initial position reading of the gripper########################
    theta_6, theta_5, theta_4, theta_3 = IK2(cam2aruco_x + base2gripper_x,cam2aruco_y + base2gripper_y, base2gripper_z)
    
    theta6 = (theta_6 + np.pi/2)
    theta5 = (np.pi/2 - theta_5)
    theta4 = (theta_4 - np.pi/2)
    theta3 = ( np.pi/2 - theta_3)
  
    if theta4 < 0: #Restriction on theta 4 to avoid weird movement
        theta4 = 0
    move(theta6, theta5, theta4, theta3)

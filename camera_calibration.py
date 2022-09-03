# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:18:57 2022

@author: lenovo
"""
import os
import glob
import numpy as np
import cv2 as cv
from pprint import pprint


def calibration_camera(row, col, path=None, cap_num=None, saving=False):
    """Calibrate the camera distortion

Parameter Description:
         row (int): The number of rows in the grid.
         col (int): The number of columns in the grid.
         path (string): The location to store the calibration image.
         cap_num (int): Indicates the number of the camera, usually 0 or 1
         saving (bool): Whether to save the camera matrix and distortion coefficients (.npz).
    """

    # Termination Criteria / Invalidation Criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points, such as (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((row * col, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    
    # Groups are used to store object points and image points from all images.
    obj_points = []  # The position of the 3d point in the real world.
    img_points = []  # The position of the 2d point in the picture.

    gray = None

    def _find_grid(img):
        # use parameters outside the function
        nonlocal gray, obj_points, img_points
        # Convert image to grayscale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the corners of the chessboard
        ret, corners = cv.findChessboardCorners(gray, (row, col), None)
        # If found, add processed 2d and 3d points
        if ret == True:
            obj_points.append(obj_p)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
            # Draw and display the corners found in the picture
            cv.drawChessboardCorners(img, (row, col), corners2, ret)

    # It is required that you must choose to use picture-based calibration or camera real-time capture calibration
    if path and cap_num:
        raise Exception("The parameter `path` and `cap_num` only need one.")
    # Image Calibration
    if path:
        # Get all images in the current path
        images = glob.glob(os.path.join(path, "*.jpg"))
        pprint(images)
        # Process each image obtained
        for f_name in images:
            # read images
            img = cv.imread(f_name)
            _find_grid(img)
            # Show pictures
            cv.imshow("img", img)
            # Picture display wait 0.5s
            cv.waitKey(500)
    # Camera live capture calibration
    if cap_num != None:
        # turn on the camera
        cap = cv.VideoCapture(cap_num)
        while True:
            # Read every frame of picture after the camera is turned on
            _, img = cap.read()
            # print(img)
            _find_grid(img)
            cv.imshow("img", img)
            cv.waitKey(400)
            print(len(obj_points))
            if len(obj_points) > 50:
                break
    # destroy all the windows
    cv.destroyAllWindows()
    # The camera matrix and distortion coefficients are obtained by calculating the acquired points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    print("ret: {}".format(ret))
    print("matrix:")
    pprint(mtx)
    print("distortion: {}".format(dist))
    # restore the parameters
    if saving:
        np.savez(os.path.join('D:/Bristol Robotics/Dissertation/img/', "mtx_dist.npz"), mtx=mtx, dist=dist)

    mean_error = 0
    for i in range(len(obj_points)):
        img_points_2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(img_points[i], img_points_2, cv.NORM_L2) / len(img_points_2)
        mean_error += error
    print("total error: {}".format(mean_error / len(obj_points)))

    return mtx, dist

    
#main function###################################################
path = 'D:/Bristol Robotics/Dissertation/img/'
mtx, dist = calibration_camera(9, 6, path = None, cap_num = 0, saving=True)

#camera calibration completed############################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mppimg
import glob


def camera_calibration():
    path_to_calibration = './camera_cal'
    nx = 9
    ny = 6
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    for i, image_name in enumerate(glob.glob(path_to_calibration + '/*.jpg')):
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow('Chess', img)
            # cv2.waitKey(0)
        else:
            print('not found')
    return objpoints, imgpoints


def undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)

#Undistortion test on chessboard

# objpoints, imgpoints = camera_calibration()
#
# img = cv2.imread('./camera_cal/calibration1.jpg')
# cv2.imshow('Disorted', img)
# cv2.waitKey(0)
# cv2.imshow('Undistored', undistort(img, objpoints, imgpoints))
# cv2.waitKey(0)

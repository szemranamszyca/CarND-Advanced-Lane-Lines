import camera_calibration as my_calib
import cv2

objpoints, imgpoints = my_calib.camera_calibration()

img_name = 'test_images/straight_lines1.jpg'
img = cv2.imread(img_name)
cv2.imshow('Distorted', img)
cv2.waitKey(0)
cv2.imshow('Undistorted', my_calib.undistort(img, objpoints, imgpoints ))
cv2.waitKey(0)

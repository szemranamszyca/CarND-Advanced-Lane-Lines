import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import camera_calibration as my_cc
import glob


def abs_sobel(img, orient='x', thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    else:
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*sobel/np.max(sobel))


    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def gradmag_sobel(img, sobel_kernel = 3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1 ,0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0 ,1, ksize = sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output


def direction_sobel(img, sobel_kernel = 3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_gradient_direction = np.arctan(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(abs_gradient_direction)
    binary_output[(abs_gradient_direction >= thresh[0]) & (abs_gradient_direction <= thresh[1])] = 1
    return binary_output


def color_select(img, sthresh=(0, 255), vtresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output_s = np.zeros_like(s_channel)
    binary_output_s[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    binary_output_v = np.zeros_like(v_channel)
    binary_output_v[(v_channel > vtresh[0]) & (v_channel <= vtresh[1])] = 1

    binary_output = np.zeros_like(s_channel)
    binary_output[(binary_output_s == 1) & (binary_output_v == 1)] = 1
    return binary_output

def transform(img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_size = (img.shape[1], img.shape[0])

    img_width = img_size[0]
    img_height = img_size[1]

    bottom_width_pct = .76
    bottom_width = img_width * bottom_width_pct

    top_width_pct = .08
    top_width = img_width * top_width_pct

    top_trim = .62
    bottom_trim = 0.935

    left_top_src = [img_width/2 - top_width/2, img_height * top_trim]
    right_top_src = [img_width/2 + top_width/2, img_height * top_trim]
    left_bottom_src= [img_width/2 - bottom_width/2, img_height * bottom_trim]
    right_bottom_src = [img_width/2 + bottom_width/2, img_height * bottom_trim]

    offset = img_width*.20
    left_top_dst = [offset, 0]
    right_top_dst = [img_width-offset, 0]
    left_bottom_dst = [offset, img_height]
    right_bottom_dst = [img_width-offset, img_height]

    src = np.float32([left_top_src, right_top_src, right_bottom_src, left_bottom_src])
    dst = np.float32([left_top_dst, right_top_dst, right_bottom_dst, left_bottom_dst])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


objpoints, imgpoints = my_cc.camera_calibration()

def pipeline(img):
    undistored = my_cc.undistort(img, objpoints, imgpoints)

    sobelx = abs_sobel(undistored, 'x', (50, 255))
    sobely = abs_sobel(undistored, 'y', (25, 255))
    color_binary = color_select(undistored, sthresh=(100, 255), vtresh=(50, 255))

    combined = np.zeros_like(sobelx)
    combined[((sobelx == 1) & (sobely == 1) | (color_binary == 1))] = 1

    transformed, M, Minv = transform(combined)

    return transformed, M, Minv


# objpoints, imgpoints = my_cc.camera_calibration()
# undistored = my_cc.undistort(img, objpoints, imgpoints)
#
# sobelx = abs_sobel(undistored, 'x', (50,255))
# sobely = abs_sobel(undistored, 'y', (25, 255))
#
# mag_binary = gradmag_sobel(undistored, thresh=(150,200))
# dir_binary = direction_sobel(undistored, thresh=(np.pi/4, np.pi/2))
# color_binary = color_select(undistored, sthresh=(100, 255), vtresh=(50,255))
#
#
# combined = np.zeros_like(mag_binary)
# combined[((sobelx == 1)  &  (sobely == 1) | (color_binary == 1) )] = 1
#
#
# transformed_nobinary, M_nb, Minv_mb = transform(undistored)
# transformed, M, Minv = transform(combined)
#
#
# img = cv2.imread('./test_images/test3.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# thresholded, M, Minv = pipeline(img)
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original', fontsize=50)
# ax2.imshow(thresholded, cmap='gray')
# ax2.set_title('Thresholded', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

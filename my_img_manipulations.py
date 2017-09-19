import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import camera_calibration as my_cc


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


def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def transform(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (img.shape[1], img.shape[0])

    left_top_src = [410,575]
    right_top_src = [930, 575]
    left_bottom_src= [280, 685]
    right_bottom_src = [1110, 685]

    offset = 25
    left_top_dst = [offset, offset]
    right_top_dst = [img_size[0]-offset, offset]
    left_bottom_dst = [offset, img_size[1]-offset]
    right_bottom_dst = [img_size[0]-offset, img_size[1]-offset]

    src = np.float32([left_top_src, right_top_src, right_bottom_src, left_bottom_src])
    dst = np.float32([left_top_dst, right_top_dst, right_bottom_dst, left_bottom_dst])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M



img_name = 'test_images/test1.jpg'
img = mpimg.imread(img_name)

objpoints, imgpoints = my_cc.camera_calibration()
undistored = my_cc.undistort(img, objpoints, imgpoints)


mag_binary = gradmag_sobel(undistored, thresh=(100,200))
dir_binary = direction_sobel(undistored, thresh=(np.pi/4, np.pi/2))
color_binary = hls_select(undistored, thresh=(150, 255))

combined = np.zeros_like(mag_binary)
combined[(color_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1



transformed, M = transform(combined)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(undistored)
ax1.set_title('Undistorted', fontsize=50)
ax2.imshow(transformed, cmap='gray')
ax2.set_title('Combined', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()

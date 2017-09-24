# **Advanced Lane Finding Project**

## Arkadiusz Konior - Project 4.

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calib1]: ./imgs/chess_calib1.png "Calibration example"
[calib2]: ./imgs/chess_calib2.png "Calibration example"
[calib3]: ./imgs/chess_calib3.png "Calibration example"
[calib_chess]: ./imgs/chess_undistorted.png "Undistorted chessboard"

[distortion1]: ./imgs/distortion1.png "Distortion examples 1"
[distortion2]: ./imgs/distortion2.png "Distortion examples 2"

[thresh]: ./imgs/thresh.png "Thresold examples"
[transform]: ./imgs/transform.png "Transform examples"
[slide]: ./imgs/transform.png "Windows slide"



[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code for this step is located in file *camera_calibration.py*
Using glob library, I've read all calibration images with parameters:

+ nx = 9
+ ny = 6

"Object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

Examples:

![Calibration example][calib1]
![Calibration example][calib2]
![Calibration example][calib3]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistored chessboard[calib_chess]


### Pipeline (single images)

#### 1.  Distortion-corrected image.

To undistort image, I've used:

`cv2.undistort` with parameters obtained from `cv2.calibrateCamera`. Here is example:
![Undistorted example][distortion]

#### 2 Color transforms, gradients or other methods to create a thresholded binary image

All operations for color transform could be find at file my_img_manipulations.py. I've decied to combine method (pipline function):

+ sobelx with threshold (50,255)
+ sobely with threshold (25,255)
+ saturation threshold (100, 255)
+ value threshold (50, 255)


![Thresholds][thresh]

#### 3. Perspective transform

Code for perspective transform could be found in my_img_manipulation, function is called `transform`. Values of trapezoid are calculate based on fact, that camera is at fixed position and they relative against image shape. 

```python
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
```


![Transformation example][transform]

#### 4. Identified lane-line pixels and fit their positions with a polynomial.

To find lane-line I've used suggested method - sliding windows. It could be find in file lane_findings.py. To get "starting points" for each line, I've calcucated histogram from the bottom part of image:

```python
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

Number of windows was set to 9. 

To find an polynomial based on founded pixels, I've used these functions:

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

![Windows slide result][slide]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

Radious of curvature and position of vehicle are calcuted in *calccurv_and_centerdist(...)*  function in lane_findings file.

To translate values from pixel space to meters, I've used formula:

```python
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = (
                     (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
```

Center of car was calcuate based on that equation:

```python
       l_fit_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
        r_fit_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Function *process_img(...)*  at file *process_video.py* is the place, where all functions meet :) Here's example how it's working:

![Example of processed image][example]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

Changing light conditions could be a challenge for my algorithm. To prevent it from losing a track, tuning thresholds method could be a solution. Also, I've implemented a basic functionality for windows slide method. Extending it could provide more reliable lane recognition.

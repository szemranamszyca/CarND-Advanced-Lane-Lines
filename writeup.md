# **Behavioral Cloning** 

## Arkadiusz Konior - Project 3.

---

**Advanced Lane Finding Project**

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

[calib]: ./imgs/calib.png "Calibration examples"
[calib_chess]: ./imgs/calib_chec.png "Calibration chessobard"
[distortion]: ./imgs/distortion.png "Distortion examples"
[thresh]: ./imgs/thresh.png "Thresold examples"


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

+nx = 9
+ny = 6

"Object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Calibration][calib]
![Undistored chessboard[calib_chess]


### Pipeline (single images)

#### 1.  Distortion-corrected image.

To undistort image, I've used:

`cv2.undistort` with parameters obtained from `cv2.calibrateCamera`. Here are examples:
![Undistorion example][distortion]

#### 2 Color transforms, gradients or other methods to create a thresholded binary image

All operations for color transform could be find at file my_img_manipulations.py. I've decied to combine method (pipline function):

+ sobelx with threshold (50,255)
+ sobely with threshold (25,255)
+ saturation threshold (100, 255)
+ value threshold (50, 255)


![Thresholds][thresh]

#### 3. {erspective transform and provide an example of a transformed image.

Code for perspective transform could be found in my_img_manipulation, function is called `transform`. Values of trapezoid are calculate based on fact, that camera is at fixed position. 

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

This resulted in the following source and destination points:


![Transformation example][transform]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

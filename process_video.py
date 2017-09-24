from moviepy.editor import VideoFileClip
# from IPython.display import HTML

from land_findings import find_lane
from land_findings import putdata
from land_findings import drawpolly


import matplotlib.image as mpimg
import matplotlib.pyplot as plt


img_name = 'test_images/straight_lines1.jpg'
img = mpimg.imread(img_name)

def process_img(img):
    binary_warped, Minv, left_fit, right_fit, left_curverad, right_curverad, center_dist = find_lane(img)
    polly = drawpolly(img, binary_warped, left_fit, right_fit, Minv)
    result = putdata(polly, (left_curverad + right_curverad)/2, center_dist)
    return result

#
# video_output1 = 'project_video_output.mp4'
# video_input1 = VideoFileClip('project_video.mp4')#subclip(0,2)
# processed_video = video_input1.fl_image(process_img)
# processed_video.write_videofile(video_output1, audio=False)

result = process_img(img)
plt.imshow(result)
plt.show()

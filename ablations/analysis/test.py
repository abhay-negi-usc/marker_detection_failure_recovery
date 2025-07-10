import cv2 
import numpy as np 
import sys 
from ablations.analysis.utils import * 
import matplotlib.pyplot as plt
import cv2


image_path = "/home/anegi/abhay_ws/marker_detection_failure_recovery/ablations/data/exp_sdg_20250625-144415/LBCV_segmentation_results/LBCV_segmentation_00000.png" 
# read image 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

corners = find_segmentation_four_corners(image)

# convert image to RGB for visualization
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

# overlay corners on image
for corner in corners:
    cv2.circle(image, tuple(corner), 3, (255, 0, 0), 1)    
# # show image 
# cv2.imshow("Corners", image)
# cv2.waitKey(0)
# cv2.destroyWindow("Corners")
# cv2.destroyAllWindows()

plt.imshow(image)
plt.axis("off")
plt.show()

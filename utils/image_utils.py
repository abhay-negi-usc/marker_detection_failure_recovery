import numpy as np 
import cv2 

def draw_keypoints(img_rgb, keypoints, color=(255, 0, 0), size=5):
    img_with_keypoints = img_rgb.copy()
    for keypoint in keypoints:
        x, y = keypoint.ravel() 
        cv2.circle(img_with_keypoints, (int(x), int(y)), size, color, -1)
    return img_with_keypoints
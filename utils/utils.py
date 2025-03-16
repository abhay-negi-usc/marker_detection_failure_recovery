import numpy as np 
import cv2 

def fit_parallelogram(binary_image):
    binary_image = np.array(binary_image)   
    # Ensure the input image is binary (already thresholded)
    if len(binary_image.shape) == 3:  # In case the image is a color image (unexpected)
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    parallelogram_corners = None

    for contour in contours:
        # Check if the contour is large enough to be considered
        area = cv2.contourArea(contour)
        if area < 100:  # Threshold for ignoring small contours (adjust as needed)
            continue

        # Approximate the contour to a polygon (reduce epsilon for higher accuracy)
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Reduce epsilon for more accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the contour has more than 4 points, apply convex hull and approximate again
        if len(approx) > 4:
            hull = cv2.convexHull(contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)  # Reduce epsilon for more accuracy
            approx = cv2.approxPolyDP(hull, epsilon, True)

        # Check if the polygon has 4 sides (quadrilateral)
        if len(approx) == 4:
            parallelogram_corners = approx.reshape(4, 2)
            break  # Stop after finding the first valid parallelogram
    
    return parallelogram_corners


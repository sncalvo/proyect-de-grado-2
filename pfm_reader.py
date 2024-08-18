import numpy as np
import cv2

groundtruth = cv2.imread('disp0.pfm')

# Remove infinite value to display
groundtruth[groundtruth==np.inf] = 0

# Normalize and convert to uint8
groundtruth = cv2.normalize(groundtruth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Show
cv2.imshow("groundtruth", groundtruth)
cv2.waitKey(0)
cv2.destroyAllWindows()
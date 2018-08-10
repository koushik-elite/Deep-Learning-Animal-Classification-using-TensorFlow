import cv2
import numpy as np

gray_img = cv2.imread('images/TriColor.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('GoldenGate',gray_img)

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()
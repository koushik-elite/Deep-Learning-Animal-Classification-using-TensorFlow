import cv2
import numpy as np

img_file = 'images/TriColor.png'
img = cv2.imread(img_file, cv2.IMREAD_COLOR)           # rgb
alpha_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED) # rgba
gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # grayscale

print(type(img))
print('RGB shape: ', img.shape)        # Rows, cols, channels
print('ARGB shape:', alpha_img.shape)
print('Gray shape:', gray_img.shape)
print('img.dtype: ', img.dtype)
print('img.size: ', img.size)
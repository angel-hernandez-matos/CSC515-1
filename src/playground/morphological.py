import cv2
import numpy as np



selected = input("1 for handwriting or 2 for fingerprint? ")
options = { '1': 'handwriting.jpg', '2': 'fingerprint.png' }
selected = options.get(selected, options['1'])

img = cv2.imread(selected)
(thresh, binary_img) = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY)
cv2.imshow('binary', binary_img)

(thresh, binary_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(binary_img, kernel, iterations = 1)
cv2.imshow('erosion', erosion)

(thresh, binary_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(binary_img, kernel, iterations = 1)
cv2.imshow('dilation', dilation)

(thresh, binary_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)

(thresh, binary_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

cv2.waitKey(0)

cv2.destroyAllWindows()
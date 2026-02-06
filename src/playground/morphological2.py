import cv2
import numpy as np

selected = input("1 for handwriting or 2 for fingerprint? ")
options = { '1': 'handwriting.jpg', '2': 'fingerprint.png' }
selected = options.get(selected, options['1'])

img = cv2.imread(selected)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Better threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('binary', binary)

kernel = np.ones((3,3), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)
cv2.imshow('erosion', erosion)

dilation = cv2.dilate(binary, kernel, iterations=1)
cv2.imshow('dilation', dilation)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)

closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

cv2.waitKey(0)
cv2.destroyAllWindows()

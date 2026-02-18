import cv2
import matplotlib.pyplot as plt
original = cv2.imread('family.jpg')

# to convert the image in grayscale
img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# applying different thresholding techniques on the input image
# all pixels value above 120 will be set to 255

ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV) 

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows = 2, ncols = 3, figsize = (10,10))

# when showing images in matplotlib, convert image from BGR to RGB
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Grayscale')
ax2.imshow(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
ax2.set_title('Binary Threshold')
ax3.imshow(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB))
ax3.set_title('Binary Threshold Inverted')
ax4.imshow(cv2.cvtColor(thresh3, cv2.COLOR_BGR2RGB))
ax4.set_title('Truncated Threshold')
ax5.imshow(cv2.cvtColor(thresh4, cv2.COLOR_BGR2RGB))
ax5.set_title('Set to 0')
ax6.imshow(cv2.cvtColor(thresh5, cv2.COLOR_BGR2RGB))
ax6.set_title('Set to 0 Inverted') 

plt.show()
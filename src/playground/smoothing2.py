import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = None

pic_to_use = int(input("Select 1 to use logo or 2 to use Halftone_Gaussian_blur picture or 3 to use Salt_and_pepper or 4 lena: "))

if pic_to_use == 1:
   img = cv.imread('opencv_logo.png')
   img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
elif pic_to_use == 2:
   img = img = cv.imread('Halftone_Gaussian_Blur.jpg', cv.IMREAD_GRAYSCALE)
elif pic_to_use == 3:
   img = cv.imread('Noise_salt_and_pepper.png')
else:
   img = cv.imread('lena.png')
   img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32) / 25
dst = cv.filter2D(img, -1, kernel)
blur = cv.blur(img, (5, 5))
gblur = cv.GaussianBlur(img, (5,5), 0)
median = cv.medianBlur(img, 5)
bilateralFilter = cv.bilateralFilter(img, 9, 75, 75)
laplacian = cv.Laplacian(img, cv.CV_32F)
blur2 = cv.GaussianBlur(img, (7,7), 10)
gray = cv.cvtColor(blur2, cv.COLOR_BGR2GRAY)
laplacian2 = cv.Laplacian(gray, cv.CV_32F)

titles = ['image', '2D Convolution', 'blur', 'GaussianBlur', 'Median', 'BilateralFilter', 'Laplacian Operator applied', 'Laplacian applied after blurring']
images = [img, dst, blur, gblur, median, bilateralFilter, laplacian, laplacian2]

for i in range(8):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
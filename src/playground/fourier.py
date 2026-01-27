import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# ---------- FIGURE 1: Original + Magnitude Spectrum ----------
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))

ax1[0].imshow(img, cmap='gray')
ax1[0].set_title('Input Image')
ax1[0].set_xticks([]), ax1[0].set_yticks([])

ax1[1].imshow(magnitude_spectrum, cmap='gray')
ax1[1].set_title('Magnitude Spectrum')
ax1[1].set_xticks([]), ax1[1].set_yticks([])

# ---------- High-pass filtering ----------
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

# ---------- FIGURE 2: Filtered results ----------
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))

ax2[0].imshow(img, cmap='gray')
ax2[0].set_title('Input Image')
ax2[0].set_xticks([]), ax2[0].set_yticks([])

ax2[1].imshow(img_back, cmap='gray')
ax2[1].set_title('Image after HPF')
ax2[1].set_xticks([]), ax2[1].set_yticks([])

ax2[2].imshow(img_back)
ax2[2].set_title('Result in JET')
ax2[2].set_xticks([]), ax2[2].set_yticks([])

plt.show()

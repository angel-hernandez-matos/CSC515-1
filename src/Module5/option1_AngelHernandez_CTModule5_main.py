# File: option1_AngelHernandez_CTModule5_main.py
# Written by: Angel Hernandez
# Description: Module 5 - Critical Thinking
# Requirement(s):
# A latent fingerprint is left on a surface by deposits of oils and/or perspiration from the finger. It is not usually
# visible but may be detected with special techniques such as dusting for fingerprints. In order to reduce rejection
# rates in most cases the acquired latent fingerprints have to be enhanced prior to matching to reduce the degradation,
# noise, or incompleteness. Enhancement can be achieved using morphological image processing.
#
# Acquire an image of a latent fingerprint. In OpenCV, write algorithms to process the image using morphological
# operations (dilation, erosion, opening, and closing).

import os
import sys
import subprocess

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class MorphologicalOperationsDemo:
    def __init__(self, fingerprint="fingerprint.png"):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt
        self.__fingerprint = fingerprint
        self.__kernel = np.ones((3, 3), np.uint8)
        self.__labels = ["Original", "Binary", "Erosion", "Dilation", "Opening", "Closing"]
        self.__image = cv2.imread(fingerprint)

    def process_fingerprint(self):
        images = [self.__image]
        gray = self.__cv2.cvtColor(self.__image, self.__cv2.COLOR_BGR2GRAY)
        # Let's use Otsu's method to perform automatic image thresholding - https://en.wikipedia.org/wiki/Otsu%27s_method
        _, binary = self.__cv2.threshold(gray, 0, 255, self.__cv2.THRESH_BINARY + self.__cv2.THRESH_OTSU)
        images.append(binary)
        images.append(self.__cv2.erode(binary, self.__kernel, iterations=1))
        images.append(self.__cv2.dilate(binary, self.__kernel, iterations=1))
        images.append(self.__cv2.morphologyEx(binary, self.__cv2.MORPH_OPEN, self.__kernel))
        images.append(self.__cv2.morphologyEx(binary, self.__cv2.MORPH_CLOSE, self.__kernel)      )
        fig, axes = self.__plt.subplots(2, 3, figsize=(14, 10))
        fig.canvas.manager.set_window_title("Morphology Operations for Fingerprint Enhancement - Option 1 - Critical Thinking - Module 5")
        assert len(images) == len(self.__labels), "Number of images does not equal number of labels."
        for i, (img, label) in enumerate(zip(images, self.__labels)):
            r = i // 3
            c = i % 3
            axes[r, c].imshow(img, cmap='gray')
            axes[r, c].set_title(label)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        self.__plt.tight_layout(rect=(0.05, 0, 1, 1))
        self.__plt.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        morphological_demo = MorphologicalOperationsDemo()
        morphological_demo.process_fingerprint()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 5 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
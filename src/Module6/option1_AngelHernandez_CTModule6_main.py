# File: option1_AngelHernandez_CTModule6_main.py
# Written by: Angel Hernandez
# Description: Module 6 - Critical Thinking
# Requirement(s):
# Find on the internet (or use a camera to take) three different types of images: an indoor scene, outdoor scenery,
# and a close-up scene of a single object. Implement an adaptive thresholding scheme to segment the images as
# best as you can.

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

class ThresholdingDemo:
    def __init__(self, images=[("Indoor Scene", "indoor.jpg"), ("Outdoor Scenery", "outdoor.jpg"), ("Close-up", "flower.jpg")]):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt
        self.__images = []
        self.__grey_images = []
        self.__selected_images = images
        self.__labels = ["Indoor Scene", "Outdoor Scenery", "Close-up", "Indoor after Thresholding",
                         "Outdoor after Thresholding", "Close-up after Thresholding"]

    def process_thresholding(self):
        # Let's have our dictionary with thresholding code that's going to be called based on image's index
        threshold_operation = {
            0: lambda img: (None, self.__cv2.adaptiveThreshold(img,255, self.__cv2.ADAPTIVE_THRESH_MEAN_C,  self.__cv2.THRESH_BINARY, 31,  5)), #Indoor
            1: lambda img: self.__cv2.threshold(img, 0, 255, self.__cv2.THRESH_BINARY + self.__cv2.THRESH_OTSU), # Outdoor
            2: lambda img: (None, self.__cv2.adaptiveThreshold(img, 255, self.__cv2.ADAPTIVE_THRESH_GAUSSIAN_C, self.__cv2.THRESH_BINARY, 21, 3)) #Close-up
        }

        # Let's load original images
        for l,i in self.__selected_images:
            self.__images.append(self.__cv2.cvtColor(self.__cv2.imread(i), self.__cv2.COLOR_BGR2RGB))
            self.__grey_images.append(self.__cv2.imread(i, self.__cv2.IMREAD_GRAYSCALE))

        # Let's call each threshold operation (for their corresponding image)
        for x, y in enumerate(self.__grey_images):
            _, th = threshold_operation[x](y)
            self.__images.append(th)

        fig, axes = self.__plt.subplots(2, 3, figsize=(14, 10))
        fig.canvas.manager.set_window_title("Adaptive Thresholding Scheme for Simple Objects - Option 1 - Critical Thinking - Module 6")
        assert len(self.__images) == len(self.__labels), "Number of images does not equal number of labels."
        for i, (img, label) in enumerate(zip(self.__images, self.__labels)):
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
        thresholding_demo = ThresholdingDemo()
        thresholding_demo.process_thresholding()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 6 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
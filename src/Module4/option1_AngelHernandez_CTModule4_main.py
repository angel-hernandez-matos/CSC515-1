# File: option1_AngelHernandez_CTModule4_main.py
# Written by: Angel Hernandez
# Description: Module 4 - Critical Thinking
# Requirement(s):
# Image filtering involves the application of window operations that perform useful functions, such as noise removal and image enhancement. Compare the effects of mean, median, and Gaussian filters on an image for different kernel windows.
#
# This image contains impulse noise. In OpenCV, write algorithms for this image to do the following:
#
# 1-. Apply mean, median, and Gaussian filters using a 3x3 kernel. Additionally, for Gaussian, select two different values of sigma.
# Think about how to select good values of sigma for optimal results.
# 2-. Apply mean, median, and Gaussian filters using a 5x5 kernel. For Gaussian, use the same values of sigma you selected in
# the above step.
# 3-. Apply mean, median, and Gaussian filters using a 7x7 kernel. For Gaussian, use the same values of sigma you selected in the above step.
# Output your filter results as 3 x 4 side-by-side subplots to make comparisons easy to inspect visually. That is, your
# subplot should have 3 rows (1 for each kernel size) and 4 columns (1 for each filter type, 2 for Gaussian). Be sure to
# include row and column labels

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

class RuntimeConfig:
    sigma1: float
    sigma2: float

    def __init__(self, sigma1: float = 0.5, sigma2: float = 1.0):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

class ImageFilterDemo:
    __np = None
    __cv2 = None
    __kernels = [3, 5, 7]
    __selected_image = None
    __labels = []

    def __init__(self, runtime_config):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt
        self.__runtime_config = runtime_config
        self.__labels = ["Mean Filter", "Median Filter", f"Gaussian σ={runtime_config.sigma1}", f"Gaussian σ={runtime_config.sigma2}"]

    def apply_filters_to_image(self, image="Mod4CT1.jpg"):
        self.__selected_image = image
        height,width = self.__load_image()
        self.__show_filtered_images()

    def __load_image(self):
        self.__image = self.__cv2.imread(self.__selected_image, self.__cv2.IMREAD_GRAYSCALE)
        return self.__image.shape

    def __show_filtered_images(self):
        fig, axes = self.__plt.subplots(3, 4, figsize=(14, 10))
        fig.canvas.manager.set_window_title("Mean, Median and Gaussian Filters - Option 1 - Critical Thinking - Module 4")
        for c, t in enumerate(self.__labels): axes[0, c].set_title(t, fontsize=12)
        fig.subplots_adjust(left=0.15)

        for row, k in enumerate(self.__kernels):
            # Row label
            axes[row, 0].set_ylabel(f"{k}×{k} Kernel", fontsize=12, labelpad=20)
            # Mean filter
            mean_img = self.__cv2.blur(self.__image, (k, k))
            # Median filter
            median_img = self.__cv2.medianBlur(self.__image, k)
            # Gaussian filters
            gauss1 = self.__cv2.GaussianBlur(self.__image, (k, k), self.__runtime_config.sigma1)
            gauss2 = self.__cv2.GaussianBlur(self.__image, (k, k), self.__runtime_config.sigma2)
            # Place in subplot
            images = [mean_img, median_img, gauss1, gauss2]
            for col in range(4):
                axes[row, col].imshow(images[col], cmap='gray')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        self.__plt.tight_layout(rect=(0.05, 0, 1, 1))
        self.__plt.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        img_filter_demo = ImageFilterDemo(TestCaseRunner.__input_runtime_config())
        img_filter_demo.apply_filters_to_image()

    @staticmethod
    def __input_runtime_config() -> RuntimeConfig:
        sigma1 = input("Specify value for σ (Sigma) 1 (default is 0.5):")
        sigma2 = input("Specify value for σ (Sigma) 2 (default is 1.0):")
        try:
            sigma1 = float(sigma1)
        except ValueError:
            sigma1 = 0.5
        try:
            sigma2 = float(sigma2)
        except ValueError:
            sigma2 = 1.0
        return RuntimeConfig(sigma1, sigma2)


def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 4 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
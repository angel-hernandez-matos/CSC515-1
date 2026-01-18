# File: option1_AngelHernandez_CT_main.py
# Written by: Angel Hernandez
# Description: Module 2 - Critical Thinking
# Requirement(s):
# 1. Import this image (using the link) into OpenCV and write code to extract each of these channels
# separately to create 2D images. This means that from the n x n x 3 shaped image, you will get 3 matrices
# of the shape n x n.
#
# 2. Now, write code to merge all these images back into a colored 3D image.
#
# 3. What will the image look like if you exchange the reds with the greens?
# Write code to merge the 2D images created in step 1 back together, this time swapping out the red
# channel with the green channel (GRB).

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

class ImageProcessor:
    __np = None
    __cv2 = None
    __selected_image = None

    def __init__(self):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt

    def process_image(self, image="shutterstock215592034--250.jpg"):
        self.__selected_image = image
        height,width,channels = self.__load_image()
        b,g,r = self.__extract_channels()
        rgb_split = self.__rgb_split(height,width,b, g, r)
        matrices = [("Original", self.__image), ("Extracted Channels", rgb_split)]
        self.__show_with_matplotlib(matrices)

    def __load_image(self):
        self.__image = self.__cv2.imread(self.__selected_image)
        return self.__image.shape

    def __extract_channels(self):
        b, g, r = self.__cv2.split(self.__image)
        return b, g, r

    def __rgb_split(self, height, width, b, g, r):
        retval = self.__np.empty([height,width*3,3], dtype=self.__np.uint8)
        retval[:, 0:width] = self.__cv2.merge([b,b,b])
        retval[:, width:width * 2] = self.__cv2.merge([g, g, g])
        retval[:, width*2:width*3] = self.__cv2.merge([r, r, r])
        return retval

    def __show_with_matplotlib(self, matrices):
        idx = 0
        dpi = 100  # typical DPI
        largest_image = next(m for n, m in matrices if n == "Extracted Channels")
        temp_height,width,channels = self.__image.shape
        temp_width = len(largest_image[1])
        tile_width = temp_width / dpi
        tile_height = temp_height / dpi
        figure = self.__plt.figure(figsize=(tile_width, tile_height), dpi=dpi) # Let's define tile's size
        figure.canvas.manager.set_window_title('Channels Extraction Demo - Critical Thinking - Module 2')
        for n,m in matrices:
            idx += 1
            self.__plt.subplot(2, 1, idx)
            self.__plt.imshow(m, cmap="gray")
            self.__plt.title(n)
            self.__plt.axis('off')
        self.__plt.tight_layout()
        self.__plt.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        img_processor = ImageProcessor()
        img_processor.process_image()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 2 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
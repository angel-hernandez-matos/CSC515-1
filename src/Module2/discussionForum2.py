# File: discussionForum2.py
# Written by: Angel Hernandez
# Description: Module 2 - Discussion Forum
# Requirement(s): Process image (banknotes), apply transformations and examine pixels matrix

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
    __plt = None
    __width = None
    __image = None
    __height = None
    __matrices = []
    __gray_image = None
    __selected_image = None

    def __init__(self):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt

    def process_image(self, image="shutterstock227361781--125.jpg"):
        self.__selected_image = image
        self.__load_image()

    def __load_image(self):
        self.__image = self.__cv2.imread(self.__selected_image)
        self.__gray_image = self.__cv2.cvtColor(self.__image, self.__cv2.COLOR_BGR2GRAY)
        self.__height, self.__width = self.__gray_image.shape
        translated = self.__translate_image()
        rotated = self.__rotate_image()
        scaled = self.__scale_image()
        perspective = self.__do_perspective_transform()
        self.__matrices = [("Original", self.__image), ("Gray", self.__gray_image), ("Translated", translated),
                    ("Rotated 90 Degrees", rotated), ("Scaled", scaled), ("Perspective", perspective)]
        how_to_show =  int(input("Press 1 to show image with OpenCV or 2 to use MatplotLib: "))
        self.__dump_matrices(self.__matrices)
        if how_to_show == 1:
            self.__show_with_opencv(self.__matrices)
        else:
            self.__show_with_matplotlib(self.__matrices)

    def __translate_image(self):
        translation = self.__cv2.resize(self.__gray_image , None, fx=1.6, fy=1.6)
        h2, w2 = translation.shape
        # Visible but not excessive shift (clear shift, minimal canvas)
        x, y = 80, 60
        temp = self.__np.float32([[1, 0, x], [0, 1, y]])
        # Canvas only slightly larger than scaled image (We cannot crop original image)
        margin = 40
        new_w = w2 + margin
        new_h = h2 + margin
        # Canvas with white background
        retval = self.__cv2.warpAffine(translation, temp, (new_w, new_h), borderValue=255)
        return retval

    def __rotate_image(self):
        angle = 90 # To rotate image 90 degrees anti-clockwise
        rotation = self.__cv2.getRotationMatrix2D((self.__width / 2, self.__height / 2), angle, 1.0)
        # Compute new bounding box to avoid cropping
        cos = self.__np.abs(rotation[0, 0])
        sin = self.__np.abs(rotation[0, 1])
        new_w = int((self.__height * sin) + (self.__width * cos))
        new_h = int((self.__height * cos) + (self.__width * sin))
        # Adjust rotation matrix to center the rotated image
        rotation[0, 2] += (new_w / 2) - self.__width / 2
        rotation[1, 2] += (new_h / 2) - self.__height / 2
        retval = self.__cv2.warpAffine(self.__gray_image, rotation, (new_w, new_h), borderValue=255)
        return retval

    def __scale_image(self):
        # Let's scale it to 3X
        return self.__cv2.resize(self.__gray_image, None, fx=3.0, fy=3.0)

    def __do_perspective_transform(self):
        perspective1 = self.__np.float32([
            [0, 0],
            [self.__width, 0],
            [0, self.__height],
            [self.__width, self.__height]
        ])

        perspective2 = self.__np.float32([
            [40, 80],  # top-left pulled inward
            [self.__width - 120, 20],  # top-right pulled down
            [20, self.__height - 60],  # bottom-left pushed right
            [self.__width - 180, self.__height]  # bottom-right pulled inward
        ])

        transform = self.__cv2.getPerspectiveTransform(perspective1, perspective2)
        retval = self.__cv2.warpPerspective(self.__gray_image, transform, (self.__width, self.__height), borderValue=255)
        return retval

    def __show_with_opencv(self, matrices):
        for n,m in matrices:
            self.__cv2.imshow(n, m)
        self.__cv2.waitKey(0)
        self.__cv2.destroyAllWindows()

    def __show_with_matplotlib(self, matrices):
        idx = 0
        dpi = 100  # typical DPI
        temp_height, temp_width =  next(m for n, m in matrices if n == "Scaled").shape
        tile_width = temp_width / dpi
        tile_height = temp_height / dpi
        figure = self.__plt.figure(figsize=(tile_width, tile_height), dpi=dpi) # Let's define tile's size
        figure.canvas.manager.set_window_title('Transformations Demo - Discussion Forum - Module 2')
        for n,m in matrices:
            idx += 1
            self.__plt.subplot(2, 3, idx)
            self.__plt.imshow(m, cmap="gray")
            self.__plt.title(n)
            self.__plt.axis('off')
        self.__plt.tight_layout()
        self.__plt.show()

    @staticmethod
    def __dump_matrices(matrices):
        i = 0
        print(f"\nCurrent Matrices: {len(matrices)}")
        for n,m in matrices:
            i += 1
            print(f"\nMatrix:{i} - {n}")
            print(f"Shape:{m.shape} - Data type:{m.dtype}")
            print(m)

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
        print('*** Module 2 - Discussion Forum ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
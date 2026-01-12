# File: mainPM1_Option1.py
# Written by: Angel Hernandez
# Description: Module 1 - Portfolio Milestone 1
# Requirement(s): Installing OpenCV 2

import os
import sys
import subprocess
import ctypes
from pathlib import Path

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class TestCaseRunner:
    @staticmethod
    def run_test():
        import numpy as np
        import cv2
        folder = None
        img = cv2.imread('shutterstock93075775--250.jpg')

        if os.name == 'nt':
            CSIDL_DESKTOP = 0
            SHGFP_TYPE_CURRENT = 0
            buf = ctypes.create_unicode_buffer(260)
            ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOP, None, SHGFP_TYPE_CURRENT, buf)
            desktop = Path(buf.value)
            folder = desktop / "CSC515-1"
        else:
            desktop = Path.home() / "Desktop"
            folder = desktop / "CSC515-1"

        folder.mkdir(exist_ok=True)
        cv2.imwrite( f"{folder}/shutterstock93075775--250_copy.jpg", img)
        cv2.imshow('shutterstock93075775--250.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        DependencyChecker.ensure_package('numpy')
        DependencyChecker.ensure_package('opencv-python')
        clear_screen()
        print('*** Module 1 - Portfolio Milestone 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
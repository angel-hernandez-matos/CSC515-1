# File: mainPM2_Option1.py
# Written by: Angel Hernandez
# Description: Module 3 - Portfolio Milestone Assignment
# Requirement(s): Option 1 - Use a camera to take a picture of yourself facing the frontal.
# In OpenCV, draw on the image a red bounding box for your eyes and a green circle around your face.
# Then tag the image with the text “this is me”.

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

class CameraCapture:
    __np = None
    __cv2 = None

    def __init__(self):
        import cv2
        import numpy as np
        self.__np = np
        self.scale = 0.5
        self.__cv2 = cv2
        self.__gray = None
        self.__resized = None
        self.__picture = None
        self.__roi_small = None

    def take_picture(self):
        self.__setup_windows()
        print("Taking picture... Press ENTER or ESCAPE to take a picture.")
        cap = self.__cv2.VideoCapture(0)  # Use the first camera
        while True:
            _, frame = cap.read()
            self.__cv2.imshow("Picture", frame)
            key = self.__cv2.waitKey(5) & 0xFF
            if key == 27 or key == 13: # Let's break if we press ENTER or ESCAPE
                self.__picture = frame
                self.__resized = self.__cv2.resize(frame, None, fx=self.scale, fy=self.scale) # If picture is too big OpenCV is likely to fail
                self.__cv2.imshow("Resized", self.__resized)
                self.__cv2.moveWindow("Resized", 0, 0)
                self.__gray = self.__cv2.cvtColor(self.__resized, self.__cv2.COLOR_BGR2GRAY)
                self.__cv2.imshow("Grayscale", self.__gray)
                self.__cv2.moveWindow("Grayscale", 350, 0)
                self.__roi_small = self.__cv2.resize(self.__gray, None, fx=0.25, fy=0.25)
                print("ROI shape:", self.__gray.shape)
                cap.release()  # Let's release the camera
                break
        self.__process_face()
        self.__cv2.destroyAllWindows()

    def __process_face(self):
        cascades = [("Faces", self.__cv2.CascadeClassifier(self.__cv2.data.haarcascades + "haarcascade_frontalface_default.xml")),
                    ("Eyes", self.__cv2.CascadeClassifier(self.__cv2.data.haarcascades + "haarcascade_eye.xml"))]

        # Let's check if the cascades we successfully loaded
        for c, d in cascades:
            if d.empty(): print(f"Error loading {c} cascade..")

        _, face_cascade = cascades[0]
        _, eye_cascade = cascades[1]
        faces = face_cascade.detectMultiScale(self.__gray, 1.1, 4, minSize=(60, 60))
        print("Faces found:", faces)
        cloned = self.__picture.copy()  # draw on original image

        for (x, y, w, h) in faces:
            # Scale coordinates back to original size
            X = int(x / self.scale); Y = int(y / self.scale);  W = int(w / self.scale);  H = int(h / self.scale)
            # Draw face circle
            center = (X + W // 2, Y + H // 2)
            radius = int(max(W, H) * 0.65)  # was 0.50 → now 30% bigger
            self.__cv2.circle(cloned, center, radius, (0, 255, 0), 2)
            # ROI on original grayscale image
            roi_gray = self.__cv2.cvtColor(self.__picture, self.__cv2.COLOR_BGR2GRAY)[Y:Y + H, X:X + W]
            # Shrink ROI for eye detection
            self.__cv2.imshow("Grayscale - ROI", roi_gray)
            self.__cv2.moveWindow("Grayscale - ROI", 700, 0)
            roi_small = self.__cv2.resize(roi_gray, None, fx=0.5, fy=0.5)
            eyes_small = eye_cascade.detectMultiScale(roi_small, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10), maxSize=(60, 60))
            print("Eyes small:", eyes_small)

            # Scale back eye coordinates
            for (ex, ey, ew, eh) in eyes_small:
                ex *= 2; ey *= 2; ew *= 2; eh *= 2
                self.__cv2.rectangle(cloned, (X + ex, Y + ey), (X + ex + ew, Y + ey + eh), (0, 0, 255), 2)
        self.__cv2.imshow("Me, myself and I", self.__tag_picture(cloned))
        self.__cv2.moveWindow("Me, myself and I", 0, 400)
        self.__cv2.waitKey(0)

    def __tag_picture(self, image):
        tag_message = "This is me"
        font = self.__cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 255, 0)  # green (BGR)
        # Get text size
        (text_width, text_height), baseline = self.__cv2.getTextSize(tag_message, font, font_scale, thickness)
        # Compute centered position
        img_h, img_w = image.shape[:2]
        x = (img_w - text_width) // 2
        y = img_h - 20  # 20px above bottom edge
        # Draw text
        self.__cv2.putText(image, tag_message,(x, y), font, font_scale, color, thickness, self.__cv2.LINE_AA)
        return image

    def __setup_windows(self):
        windows = ["Picture", "Resized", "Grayscale", "Grayscale - ROI", "Me, myself and I"]

        for w in windows:
            self.__cv2.namedWindow(w, self.__cv2.WINDOW_NORMAL)

class TestCaseRunner:
    @staticmethod
    def run_test():
        cam_capture = CameraCapture()
        cam_capture.take_picture()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 3 - Portfolio Milestone Assignment ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
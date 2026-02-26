# File: AHernandez_PPModule8_Option1_main.py
# Written by: Angel Hernandez
# Description: Module 8 - Portfolio Project
#
# Requirement(s):
#
#  The goal of this project is to write algorithms for license plate detection and license plate character recognition.
#  Select three color images from the internet that meet the following requirements:
#
# Two images containing vehicles with Russian license plates and one image of vehicles with a non-Russian plate.
# All images should include the entire vehicle and not just the license plate.
# At least one image with Russian plates should display the license plate far away.
# At least one image should include multiple vehicles.
# All images should vary in light illumination and color intensity.
# First, using the appropriate trained cascade classifier., write one algorithm to detect the Russian
# license plate in the gray scaled versions of the original images.  Put a red boundary box around the detected plate
# in the image in order to see what region the classifier deemed as a license plate.  If expected results are not
# achieved on the unprocessed images, apply processing steps before implementing the classifier for optimal results.
#
# After the license plates have been successfully detected, you will want to process only the extracted plate region
# before applying character recognition on it.  Although the license plate number classifier.
# is fairly accurate, it is important that all license plates are rotated and scaled so that they are horizontally aligned.
# If expected results are not achieved, implement more image processing for optimal character recognition.

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

class LicensePlateDetector:
    def __init__(self, images=[("Car with Russian Plates", "car_with_russian_plates.jpg"),
                               ("Multiple Russian Cars in Traffic", "multiple_russian_cars_in_traffic.jpg"),
                               ("European Cars", "cars_in_europe.jpg")]):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt
        self.__images = []
        self.__output_dir = "output"
        self.__selected_images = images
        self.__cascade = None
        self.__cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        self.__labels = ["Car with Russian Plates", "Multiple Russian Cars in Traffic", "European Cars",
                         "Car with Russian Plates (Post-Detection)",
                         "Multiple Russian Cars in Traffic (Post-Detection)",
                         "European Cars (Post-Detection)"]
        os.makedirs(self.__output_dir, exist_ok=True)
        self.__load_cascade()

    def __load_cascade(self):
        self.__cascade = self.__cv2.CascadeClassifier(self.__cascade_path)
        if self.__cascade.empty():
            raise FileNotFoundError(f"Cascade file not found in {self.__cascade_path}")

    def __preprocess_for_detection(self, gray_img):
        clahe = self.__cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(gray_img)
        retval = self.__cv2.bilateralFilter(norm, 9, 75, 75)
        return retval

    def __detect_plates(self, image_bgr, cascade):
        gray_img = self.__cv2.cvtColor(image_bgr, self.__cv2.COLOR_BGR2GRAY)
        proc = self.__preprocess_for_detection(gray_img)

        plates = cascade.detectMultiScale(proc, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=self.__cv2.CASCADE_SCALE_IMAGE)
        return plates, gray_img

    def __draw_plate_boundaries(self, image_bgr, plates):
        retval = image_bgr.copy()
        for (x, y, w, h) in plates:
            self.__cv2.rectangle(retval, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return retval

    @staticmethod
    def __extract_plate_region(gray_img, plate_bbox, margin=0.05):
        x, y, w, h = plate_bbox
        dx = int(w * margin)
        dy = int(h * margin)
        x0 = max(x - dx, 0)
        y0 = max(y - dy, 0)
        x1 = min(x + w + dx, gray_img.shape[1])
        y1 = min(y + h + dy, gray_img.shape[0])
        return gray_img[y0:y1, x0:x1]

    def __deskew_and_resize(self, gray_plate, target_size=(200, 50)):
        blur = self.__cv2.GaussianBlur(gray_plate, (5, 5), 0)
        _, th = self.__cv2.threshold(blur, 0, 255, self.__cv2.THRESH_BINARY + self.__cv2.THRESH_OTSU)

        if self.__np.mean(th) > 127:
            th = 255 - th

        if (coords := self.__cv2.findNonZero(th)) is None:
            return self.__cv2.resize(gray_plate, target_size)

        rect = self.__cv2.minAreaRect(coords)
        (cx, cy), (w, h), angle = rect

        if w < h:
            angle += 90.0

        m = self.__cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = self.__cv2.warpAffine(gray_plate, m, (gray_plate.shape[1], gray_plate.shape[0]),
                                          flags=self.__cv2.INTER_CUBIC, borderMode=self.__cv2.BORDER_REPLICATE)

        box = self.__cv2.boxPoints(rect).astype(int)
        xs = box[:, 0]
        ys = box[:, 1]
        x0, x1 = max(xs.min(), 0), min(xs.max(), rotated.shape[1] - 1)
        y0, y1 = max(ys.min(), 0), min(ys.max(), rotated.shape[0] - 1)
        cropped = rotated[y0:y1, x0:x1]

        if cropped.size == 0:
            cropped = gray_plate

        return self.__cv2.resize(cropped, target_size)

    def process_images(self):
        detected = []
        russian_cars = ["car_with_russian_plates.jpg", "multiple_russian_cars_in_traffic.jpg"]
        for img_path in russian_cars:
            self.__images.append(self.__cv2.cvtColor(self.__cv2.imread(img_path), self.__cv2.COLOR_BGR2RGB))
            detected.append(self.__process_image_for_plates(img_path, prefix="russian"))
        self.__images.append(self.__cv2.cvtColor(self.__cv2.imread("cars_in_europe.jpg"), self.__cv2.COLOR_BGR2RGB))
        detected.append(self.__process_image_for_plates("cars_in_europe.jpg", prefix="european"))
        self.__images += detected
        self.__show_images()

    def __process_image_for_plates(self, image_path, prefix=""):
        print(f"\n===> Processing {image_path} <===")
        img = self.__cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read {image_path}")
            return None

        plates, gray =  self.__detect_plates(img, self.__cascade)
        print(f"Detected {len(plates)} plate(s)\n")

        boxed = self.__draw_plate_boundaries(img, plates)
        boxed_path = os.path.join(self.__output_dir, f"{prefix}_boxed_{os.path.basename(image_path)}")
        self.__cv2.imwrite(boxed_path, boxed)
        retval = self.__cv2.cvtColor(self.__cv2.imread(boxed_path), self.__cv2.COLOR_BGR2RGB)
        print(f"Saved boxed image to {boxed_path}")

        for i, boundary in enumerate(plates):
            print(f"\n[ Plate {i + 1} ]")
            print(f"boundary: {boundary}")
            plate_region = self.__extract_plate_region(gray, boundary)
            plate =  self.__deskew_and_resize(plate_region)
            plate_out = os.path.join(self.__output_dir, f"{prefix}_plate_{i}_{os.path.basename(image_path)}")
            self.__cv2.imwrite(plate_out, plate)
            print(f"Saved extracted plate: {plate_out}")
        return retval

    def __show_images(self):
        fig, axes = self.__plt.subplots(2, 3, figsize=(14, 10))
        fig.canvas.manager.set_window_title("License Plate Detection and Faked Content - Option 1 - Portfolio Project - Module 8")
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
        license_plate_instance = LicensePlateDetector()
        license_plate_instance.process_images()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 8 - Portfolio Project ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
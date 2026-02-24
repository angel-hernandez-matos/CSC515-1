# File: discussion_Forum_main.py
# Written by: Angel Hernandez
# Description: Module 7 - Critical Thinking
# Requirement(s):
# Using OpenCV, generate a synthetic image that contains exactly one filled-in square and one filled-in circle.
# The placement and color intensities of these shapes are up to you. The background intensity is up to you as well.
# You should know precisely the locations of the discontinuities. The rest of the image should be without edges.
#
# Implement Canny, Sobel, and then Laplacian edge detection on this image. Define a measure to evaluate the performance
# of each method. Repeat this experiment by adding noise to the image using a random number generator and changing the
# intensity values of the objects and the background. Change the threshold values for your detection step and then
# evaluate the performance once again.

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


class ImageCreationParam:
    def __init__(self, width=300, height=300, background_intensity=50, square_intensity=250,
                 circle_intensity=200, add_noise=False, noise_std=15):
        self.__width = width
        self.__height = height
        self.__background_intensity = background_intensity
        self.__square_intensity = square_intensity
        self.__add_noise = add_noise
        self.__noise_std = noise_std
        self.__circle_intensity = circle_intensity

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        if not value: raise ValueError("width cannot be empty")
        self.__width = value

    @width.deleter
    def width(self):
        del self.__width

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        if not value: raise ValueError("height cannot be empty")
        self.__height = value

    @height.deleter
    def height(self):
        del self.__height

    @property
    def background_intensity(self):
        return self.__background_intensity

    @background_intensity.setter
    def background_intensity(self, value):
        if not value: raise ValueError("background_intensity cannot be empty")
        self.__background_intensity = value

    @background_intensity.deleter
    def background_intensity(self):
        del self.__background_intensity

    @property
    def square_intensity(self):
        return self.__square_intensity

    @square_intensity.setter
    def square_intensity(self, value):
        if not value: raise ValueError("square_intensity cannot be empty")
        self.__square_intensity = value

    @square_intensity.deleter
    def square_intensity(self):
        del self.__square_intensity

    @property
    def circle_intensity(self):
        return self.__circle_intensity

    @circle_intensity.setter
    def circle_intensity(self, value):
        if not value: raise ValueError("circle_intensity cannot be empty")
        self.__circle_intensity = value

    @circle_intensity.deleter
    def circle_intensity(self):
        del self.__circle_intensity

    @property
    def add_noise(self):
        return self.__add_noise

    @add_noise.setter
    def add_noise(self, value):
        if not value: raise ValueError("add_noise cannot be empty")
        self.__add_noise = value

    @add_noise.deleter
    def add_noise(self):
        del self.__add_noise

    @property
    def noise_std(self):
        return self.__noise_std

    @noise_std.setter
    def noise_std(self, value):
        if not value: raise ValueError("noise_std cannot be empty")
        self.__noise_std = value

    @noise_std.deleter
    def noise_std(self):
        del self.__noise_std


class EdgeDetectionDemo:
    def __init__(self, img_creation_param=None):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        if img_creation_param is None:
            img_creation_param = ImageCreationParam()

        self.__np = np
        self.__cv2 = cv2
        self.__plt = plt
        self.__synthetic_image = None
        self.__image_creation_param = img_creation_param
        self.tiny_number = 1e-8  # To prevent division by zero
        self.__images = { "sobel": [], "laplacian": [], "canny": [] }

    def __create_synthetic_image(self):
        # Let's create image
        h, w = (self.__image_creation_param.width, self.__image_creation_param.height)
        img = self.__np.full((h, w), self.__image_creation_param.background_intensity, dtype=self.__np.uint8)
        self.__synthetic_image = img
        # Top-left (80, 80) and bottom-right (150, 150)
        square_tl = (80, 80)
        square_br = (150, 150)
        # Define circle (known center and radius) # e.g., center (180, 180), radius 30
        circle_radius = 30
        circle_center = (180, 180)
        # Draw filled square and circle
        self.__cv2.rectangle(img, square_tl, square_br, self.__image_creation_param.square_intensity, thickness=-1)
        self.__cv2.circle(img, circle_center, circle_radius, self.__image_creation_param.circle_intensity, thickness=-1)

        if self.__image_creation_param.add_noise:
            noise = self.__np.random.normal(0, self.__image_creation_param.noise_std, img.shape).astype(self.__np.float32)
            noisy = img.astype(self.__np.float32) + noise
            noisy = self.__np.clip(noisy, 0, 255).astype(self.__np.uint8)
            img = noisy

        return img, square_tl, square_br, circle_center, circle_radius

    def __create_edge_map(self, img_size, square_tl, square_br, circle_center, circle_radius):
        h, w = img_size
        mask = self.__np.zeros((h, w), dtype=self.__np.uint8)
        # Binary mask
        self.__cv2.rectangle(mask, square_tl, square_br, 255, thickness=-1)
        self.__cv2.circle(mask, circle_center, circle_radius, 255, thickness=-1)
        # Morphological gradient
        kernel = self.__np.ones((3, 3), self.__np.uint8)
        retval = self.__cv2.morphologyEx(mask, self.__cv2.MORPH_GRADIENT, kernel)
        retval = (retval > 0).astype(self.__np.uint8)

        return retval

    def __sobel_edge_detector(self, img, kernel_size=3):
        # Compute gradient magnitude
        gx = self.__cv2.Sobel(img, self.__cv2.CV_64F, 1, 0, ksize=kernel_size)
        gy = self.__cv2.Sobel(img, self.__cv2.CV_64F, 0, 1, ksize=kernel_size)
        magnitude = self.__np.sqrt(gx**2 + gy**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(self.__np.uint8)
        return magnitude

    def __laplacian_edge_detector(self, img, kernel_size=3):
        laplacian = self.__cv2.Laplacian(img, self.__cv2.CV_64F, ksize=kernel_size)
        lap_abs = self.__np.abs(laplacian)
        lap_abs = (lap_abs / lap_abs.max() * 255).astype(self.__np.uint8)
        return lap_abs

    def __canny_edges(self, img, low_thresh, high_thresh):
        return self.__cv2.Canny(img, low_thresh, high_thresh)

    def __evaluate_edges(self, pred_edges, gt_edges):
        predicted_edges = (pred_edges > 0).astype(self.__np.uint8)
        ground_truth = (gt_edges > 0).astype(self.__np.uint8)
        true_positives = self.__np.logical_and(predicted_edges == 1, ground_truth == 1).sum()
        false_positives = self.__np.logical_and(predicted_edges == 1, ground_truth == 0).sum()
        false_negatives = self.__np.logical_and(predicted_edges == 0, ground_truth == 1).sum()
        precision = true_positives / (true_positives + false_positives + self.tiny_number)
        recall = true_positives / (true_positives + false_negatives + self.tiny_number)
        f1 = 2 * precision * recall / (precision + recall + self.tiny_number)
        return { "TruePositives": int(true_positives), "FalsePositives": int(false_positives),
                 "FalseNegatives": int(false_negatives), "Precision": float(precision),
                 "Recall": float(recall), "F1": float(f1)}

    def __run_scenario(self, add_noise=False, noise_std=10, bg_intensity=30, square_intensity=200,
                     circle_intensity=150, sobel_thresh_list=(50, 100, 150), lap_thresh_list=(20, 40, 60),
                     canny_thresh_pairs=((50, 150), (100, 200), (150, 250))):

        img, sq_tl, sq_br, c_center, c_radius = self.__create_synthetic_image()
        ground_truth = self.__create_edge_map(img.shape, sq_tl, sq_br, c_center, c_radius)

        # Sobel
        sobel_results = {}
        sobel_magnitude = self.__sobel_edge_detector(img)
        for th in sobel_thresh_list:
            _, sobel_bin = self.__cv2.threshold(sobel_magnitude, th, 255, self.__cv2.THRESH_BINARY)
            sobel_bin = (sobel_bin > 0).astype(self.__np.uint8)
            sobel_results[th] = self.__evaluate_edges(sobel_bin, ground_truth)
        self.__images["sobel"].append(sobel_magnitude)

        # Laplacian
        lap_results = {}
        lap_magnitude = self.__laplacian_edge_detector(img)
        for th in lap_thresh_list:
            _, lap_bin = self.__cv2.threshold(lap_magnitude, th, 255, self.__cv2.THRESH_BINARY)
            lap_bin = (lap_bin > 0).astype(self.__np.uint8)
            lap_results[th] = self.__evaluate_edges(lap_bin, ground_truth)
        self.__images["laplacian"].append(lap_magnitude)

        # Canny
        canny_results = {}
        for low, high in canny_thresh_pairs:
            canny = self.__canny_edges(img, low, high)
            canny_bin = (canny > 0).astype(self.__np.uint8)
            canny_results[(low, high)] = self.__evaluate_edges(canny_bin, ground_truth)
        self.__images["canny"].append(canny_bin)

        return { "sobel": sobel_results, "laplacian": lap_results, "canny": canny_results }

    @staticmethod
    def __print_tables(results, title="Scenario"):
        def __print_helper(detector_name, data, is_canny=False):
            print(f"\n{detector_name}")
            if is_canny:
                print("╔═════════════════════════════════════════════════════════════════════════════════════════╗")
                print("║ Low–High ║ True Positives ║ False Positives ║ False Negatives ║ Precision ║ Recall ║ F1 ║")
                print("║══════════║════════════════║═════════════════║═════════════════║═══════════║════════║════║")
                for (low, high), metrics in data.items():
                    print(f"║ {low}–{high} ║ {metrics['TruePositives']} ║ {metrics['FalsePositives']} ║ "
                          f"{metrics['FalseNegatives']} ║ {metrics['Precision']:.4f} ║ "
                          f"{metrics['Recall']:.4f} ║ {metrics['F1']:.4f} ║")
            else:
                print("╔══════════════════════════════════════════════════════════════════════════════════════════╗")
                print("║ Threshold ║ True Positives ║ False Positives ║ False Negatives ║ Precision ║ Recall ║ F1 ║")
                print("║═══════════║════════════════║═════════════════║═════════════════║═══════════║════════║════║")
                for th, metrics in data.items():
                    print(f"║ {th} ║ {metrics['TruePositives']} ║ {metrics['FalsePositives']} ║ "
                          f"{metrics['FalseNegatives']} ║ {metrics['Precision']:.4f} ║ "
                          f"{metrics['Recall']:.4f} ║ {metrics['F1']:.4f} ║")

        print(f"\n===> {title} <===\n")

        __print_helper("Sobel", results["sobel"])
        __print_helper("Laplacian", results["laplacian"])
        __print_helper("Canny", results["canny"], is_canny=True)

    def run_orchestration(self):
        clean = self.__run_scenario(add_noise=False)
        self .__print_tables(clean, "Clean Image with default Intensities")

        noisy = self.__run_scenario(add_noise=True, noise_std=15)
        self.__print_tables(noisy, "Noisy Image with default Intensities")

        changed = self.__run_scenario(add_noise=False, bg_intensity=80, square_intensity=220,
                                          circle_intensity=40, sobel_thresh_list=(30, 60, 90),
                                          lap_thresh_list=(10, 30, 50),
                                          canny_thresh_pairs=((30, 100), (80, 160), (120, 220)))

        self.__print_tables(changed, "Clean image with changes to intensities and thresholds")
        self.__show_images()

    def __show_group(self, fig, start_row, group_name, title_prefix):
        plt = self.__plt
        images = self.__images.get(group_name, [])
        cols = 3  # choose any layout you like
        rows = (len(images) + cols - 1) // cols

        for i, img in enumerate(images):
            ax = fig.add_subplot(start_row + i // cols, cols, (i % cols) + 1)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{title_prefix} #{i + 1}")
            ax.axis("off")

        return rows

    def __show_images(self):
        cols = 3
        plt = self.__plt
        images = [("Synthetic Image", self.__synthetic_image)]
        for name in ["sobel", "laplacian", "canny"]:
            for i, img in enumerate(self.__images[name]):
                images.append((f"{name.capitalize()} #{i + 1}", img))

        total = len(images)
        rows = (total + cols - 1) // cols  # ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, images):
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        # Hide any unused axes
        for ax in axes[len(images):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        edge_detection_demo = EdgeDetectionDemo()
        edge_detection_demo.run_orchestration()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'opencv-python', 'matplotlib']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 7 - Discussion Forum ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()
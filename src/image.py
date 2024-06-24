import os
import cv2
import numpy as np
from skimage.feature import daisy, hog
from scipy.stats import skew


class img:
    def __init__(
        self, path="", name="", standard_size=(64, 64), train=True, window=None
    ):
        self.data = []
        self.skip = False
        self.path = path
        self.name = name
        self.label = self.get_label_from_name()
        self.standard_size = standard_size
        self.train = train
        self.window = window
        self.preprocess()

    def get_label_from_name(self):
        parts = self.name.split("_")
        try:
            cropped_index = parts.index("cropped")
            return parts[cropped_index + 1]
        except (ValueError, IndexError):
            # print(f"Warning: {self.name} could not be read.")
            return "none"

    @staticmethod
    def color_preprocess(img):
        # Convert the image to LAB color space for better color normalization
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Apply histogram equalization to the L channel
        l_channel = cv2.equalizeHist(l_channel)

        # Merge the LAB channels back
        img_lab = cv2.merge((l_channel, a_channel, b_channel))

        # Convert back to BGR color space
        img_normalized = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        return cv2.split(img_normalized)

    @staticmethod
    def compute_color_moments(image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Compute the moments for each channel
        moments = []
        for channel in cv2.split(hsv_image):
            mean = np.mean(channel)
            std = np.std(channel)
            # Handle cases where standard deviation is zero
            if std == 0:
                skewness = 0
            else:
                skewness = skew(channel.flatten())
            # Check for NaN or infinite values and handle them
            if np.isnan(skewness) or np.isinf(skewness):
                skewness = 0
            moments.extend([mean, std, skewness])
        return np.array(moments)

    @staticmethod

    @staticmethod
    def compute_hog_feature(image):
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bins = 9
        cell_size = (8, 8)
        cpb = (2, 2)
        norm = "L2"
        hog_feature = hog(
            image,
            orientations=bins,
            pixels_per_cell=cell_size,
            cells_per_block=cpb,
            block_norm=norm,
            transform_sqrt=True,
        ).flatten()
        return hog_feature

    @staticmethod
    def compute_feu_color(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define color ranges
        # Red color range
        lower_red = np.array([150, 0, 0])
        upper_red = np.array([255, 100, 100])
        # Green color range
        lower_green = np.array([0, 150, 0])
        upper_green = np.array([100, 255, 100])

        # Orange color range
        lower_orange = np.array([200, 100, 0])
        upper_orange = np.array([255, 170, 100])

        # Create masks
        red_mask = cv2.inRange(image_rgb, lower_red, upper_red)
        green_mask = cv2.inRange(image_rgb, lower_green, upper_green)
        orange_mask = cv2.inRange(image_rgb, lower_orange, upper_orange)

        # Count the number of pixels for each color
        red_pixels = np.sum(red_mask > 0)
        green_pixels = np.sum(green_mask > 0)
        orange_pixels = np.sum(orange_mask > 0)
        other_pixels = image.shape[0] * image.shape[1] - red_pixels - green_pixels - orange_pixels

        #normalize the values
        red_pixels /= image.shape[0] * image.shape[1]
        green_pixels /= image.shape[0] * image.shape[1]
        orange_pixels /= image.shape[0] * image.shape[1]
        other_pixels /= image.shape[0] * image.shape[1]

        return np.array([red_pixels, green_pixels, orange_pixels, other_pixels])

    @staticmethod
    def compute_brightness(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2]
        summed_brightness = np.sum(brightness, axis=1)

        max_brightness = np.max(summed_brightness)
        if max_brightness != 0:
            summed_brightness = summed_brightness / max_brightness
        else:
            # Handle the case where the max_brightness is zero
            summed_brightness = np.zeros_like(summed_brightness)
        return summed_brightness
    @staticmethod
    def compute_brightnessv2(image,num_boxes_x, num_boxes_y):
        height, width, _ = image.shape
        box_height = height // num_boxes_y
        box_width = width // num_boxes_x
        brightness_matrix = np.zeros((num_boxes_y, num_boxes_x))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2]
        for i in range(num_boxes_y):
            for j in range(num_boxes_x):
                start_x = j * box_width
                end_x = (j + 1) * box_width if (j + 1) * box_width < width else width
                start_y = i * box_height
                end_y = (i + 1) * box_height if (i + 1) * box_height < height else height

                box_brightness = brightness[start_y:end_y, start_x:end_x]
                average_brightness = np.mean(box_brightness)
                brightness_matrix[i, j] = average_brightness
        max_brightness = np.max(brightness_matrix)
        min_brightness = np.min(brightness_matrix)
        if max_brightness != min_brightness:
            brightness_matrix = (brightness_matrix - min_brightness) / (max_brightness - min_brightness)
        else:
            brightness_matrix = np.zeros_like(brightness_matrix)
        return brightness_matrix.flatten()

    def preprocess(self):
        self.data = []
        try:
            if self.window is None:
                img = cv2.imread(self.path + self.name)
            else:
                img = self.window
            img = cv2.resize(img, self.standard_size)

            #self.data.extend(compute_daisy_feature(img))
            self.data.extend(self.compute_hog_feature(img))
            self.data.extend(self.compute_color_moments(img))
            #self.data.extend(self.compute_brightness(img))
            self.data.extend(self.compute_brightnessv2(img, 4, 4))
            #self.data.extend(self.compute_feu_color(img))
            #print(f"Shape of img.data: {len(self.data)}")

        except Exception as e:
            print(f"Error preprocessing: {e}")
            self.skip = True
            self.data = None

def compute_daisy_feature(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    daisy_feature = daisy(
        img_gray,
        step=4,
        radius=3,
        rings=2,
        histograms=8,
        orientations=16,
        normalization="l2",
    ).flatten()
    return daisy_feature

import os
import cv2
import numpy as np
from skimage.feature import daisy
from scipy.stats import skew


class img:
    def __init__(
        self, path="", name="", standard_size=(32, 32), train=True, window=None
    ):
        self.data = []
        self.skip = False
        self.path = path
        self.name = name
        self.label = self.get_label_from_name()
        self.standard_size = standard_size
        self.train = train
        self.window = window

    def get_label_from_name(self):
        parts = self.name.split("_")
        try:
            cropped_index = parts.index("cropped")
            return parts[cropped_index + 1]
        except (ValueError, IndexError):
            print(f"Warning: {self.name} could not be read.")
            return None

    def color_preprocess(self, img):
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

    def compute_color_moments(self, image):
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

    def compute_daist_feature(self, image):
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

    def preprocess(self):
        try:
            if  self.window is None:
                img = cv2.imread(self.path + self.name)
            else:
                img = self.window
            img = cv2.resize(img, self.standard_size)
            self.data.extend(self.compute_daist_feature(img))
            self.data.extend(self.compute_color_moments(img))
        except Exception as e:
            print(f"Error in preprocess: {e}")
            self.skip = True
            self.data = None


def _load_single_image(img_dir, img_name, standard_size, train):
    return img(img_dir, img_name, standard_size, train)

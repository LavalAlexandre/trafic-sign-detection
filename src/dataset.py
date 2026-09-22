import os
import random
import time

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.image import img
from src.labels import LABELS

# interdiction is heavily over-represented in the training set
DEFAULT_UNDERSAMPLE = {"interdiction": 0.25}


class dataset:
    def __init__(
            self,
            img_dir,
            standard_size=(64, 64),
            train=True,
            augment_path="",
            label_size_factor=1,
            undersample=None,
            seed=42,
    ):
        """
        undersample: {label: fraction} of images to drop for that label (training only).
                     Defaults to DEFAULT_UNDERSAMPLE; pass {} to disable.
        """
        time_start = time.time()
        self.standard_size = standard_size
        self.train = train
        self.rng = random.Random(seed)
        self.images = []  # list of img objects
        self.img_dir = img_dir
        self.load_imgs()
        # Each label is augmented up to 2/3 of the largest class, scaled by label_size_factor
        self.max_label = int(self.max_label * 2 / 3) * label_size_factor
        if augment_path != "" and train and label_size_factor > 0:
            self.augment(augment_path, self.max_label)
        self.LabelEncoder = LabelEncoder()
        self.LabelEncoder.fit(LABELS)
        if train:
            self.undersample(DEFAULT_UNDERSAMPLE if undersample is None else undersample)

        time_end = time.time()
        print(f"Total loading time: {time_end - time_start:.2f} seconds")

    def undersample(self, fractions):
        for label, fraction in fractions.items():
            label_imgs = [im for im in self.images if im.label == label]
            dropped = set(map(id, self.rng.sample(label_imgs, int(len(label_imgs) * fraction))))
            self.images = [im for im in self.images if id(im) not in dropped]

    def augment(self, augment_path, target_size):
        augment_label_map = {label: [] for label in self.label_map.keys()}
        img_names = os.listdir(augment_path)
        print(f"Found {len(img_names)} images in {augment_path}")
        # load the augmented images
        for img_name in tqdm(img_names, desc="Loading augmented images"):
            new_img = img(augment_path, img_name, self.standard_size, self.train)
            if new_img.skip:
                continue
            if new_img.label in augment_label_map:
                augment_label_map[new_img.label].append(new_img)

        for label in augment_label_map:
            print(f"Label {label} has {len(augment_label_map[label])} augmented images")

        for label in self.label_map:
            current_size = len(self.label_map[label])
            n_augment = int(target_size - current_size)
            if n_augment <= 0:
                print(f"Skipping label {label} as it already has enough images")
                continue
            if not augment_label_map.get(label):
                print(f"No augmented images found for label {label}")
                continue
            print(
                f"Augmenting label {label} with {n_augment} images, using {len(augment_label_map[label])} images"
            )
            self.augment_label(augment_label_map[label], n_augment)

    def augment_label(self, augment_imgs, n_augment):
        possible_augment = augment_imgs.copy()
        self.rng.shuffle(possible_augment)
        while n_augment > 0:
            self.images.append(possible_augment.pop())
            n_augment -= 1
            if not possible_augment:
                possible_augment = augment_imgs.copy()
                self.rng.shuffle(possible_augment)

    def load_imgs(self):
        self.label_map = {}
        img_names = sorted(os.listdir(self.img_dir))
        for img_name in tqdm(img_names, desc=f"Loading {self.img_dir}"):
            new_img = img(self.img_dir, img_name, self.standard_size, self.train)
            if new_img.skip:
                print(f"Skipping image {new_img.name}")
                continue
            self.images.append(new_img)
            self.label_map.setdefault(new_img.label, []).append(new_img)
        self.max_label = max(len(imgs) for imgs in self.label_map.values())

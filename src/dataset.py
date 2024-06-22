import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
import time
import gc
from src.image import img


class dataset:
    def __init__(
            self,
            img_dir,
            standard_size=(64, 64),
            train=True,
            augment_path="",
            label_size_factor=1,
    ):
        time_start = time.time()
        self.standard_size = standard_size
        self.train = train
        #self.data = None
        self.images = []  #list of img objects
        self.max_label = 0
        self.img_dir = img_dir
        self.load_imgs()
        self.max_label = 200
        self.max_label *= label_size_factor
        if augment_path != "" and train and label_size_factor > 0:
            self.augment(augment_path, self.max_label)  # add aug img to self.images
        self.LabelEncoder = LabelEncoder()
        self.LabelEncoder.fit(
            [
                "none",
                "interdiction",
                "danger",
                "fvert",
                "stop",
                "obligation",
                "forange",
                "ceder",
                "frouge",
            ]
        )
        gc.collect()
        # now preprocess the data
        self.preprocess()

        #remove half of the interdiction images

        interdiction_imgs = [img for img in self.images if img.label == "interdiction"]

        for i in range(len(interdiction_imgs)//2):
            self.images.remove(interdiction_imgs[i])





        time_end = time.time()
        print(f"Total loading time: {time_end - time_start:.2f} seconds")

    def preprocess(self):
        """
        Preprocess the images by applying necessary transformations.
        """
        imgs_data = []
        filtered_images = []
        for img in tqdm(self.images, desc="Preprocessing images"):
            if img.skip:
                print(f"Skipping image {img.name}")
                continue
            imgs_data.append(img.data)
            filtered_images.append(img)

        self.images = filtered_images
        #self.data = np.array(imgs_data)


    def augment(self, augment_path, target_size):


        augment_label_map = {label: [] for label in self.label_map.keys()}
        img_names = os.listdir(augment_path)
        print(f"Found {len(img_names)} images in {augment_path}")
        #load the augmented images
        for img_name in img_names:
            new_img = img(augment_path, img_name, self.standard_size, self.train)
            if new_img.skip:
                print(f"Skipping image {new_img.name}")
                continue
            else:
                augment_label_map[new_img.label].append(new_img)

        #print the size of all the augmented images
        for label in augment_label_map:
            print(f"Label {label} has {len(augment_label_map[label])} images")

        imgs = []
        for label in self.label_map:
            """
            if label == "none":
                continue
            """
            current_size = len(self.label_map[label])
            n_augment = int(target_size - current_size)
            if n_augment <= 0:
                print(f"Skipping label {label} as it already has enough images")
                continue
            if label not in augment_label_map:
                print(f"No augmented images found for label {label}")
                continue
            if len(augment_label_map[label]) == 0:
                print(f"No augmented images found for label {label}")
                continue
            print(
                f"Augmenting label {label} with {n_augment} images, using {len(augment_label_map[label])} images"
            )
            self.augment_label(augment_label_map[label], n_augment)
        self.images.extend(imgs)
        self.label_map = {}

    def augment_label(self, augment_imgs, n_augment):
        # shuffle the augment_imgs
        random.shuffle(augment_imgs)
        possible_augment = augment_imgs.copy()
        while n_augment > 0:
            self.images.append(possible_augment.pop())
            n_augment -= 1
            if not possible_augment:
                possible_augment = augment_imgs.copy()
                random.shuffle(possible_augment)

    def load_imgs(self):
        self.label_map = {}
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            new_img = img(self.img_dir, img_name, self.standard_size, self.train)
            if new_img.skip:
                continue
            self.images.append(new_img)
            if new_img.label not in self.label_map:
                self.label_map[new_img.label] = [new_img]
            else:
                self.label_map[new_img.label].append(new_img)
        max_label = max([len(self.label_map[label]) for label in self.label_map])
        self.max_label = max_label

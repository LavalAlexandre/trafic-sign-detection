import os
import shutil
import time
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from imgaug import BoundingBoxesOnImage, BoundingBox
from imgaug.augmentables.batches import UnnormalizedBatch
import imgaug.augmenters as iaa


def build_seq():
    seq = iaa.Sequential(
        [
            iaa.Crop(percent=(0, 0.1)),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8),
            ),
        ],
        random_order=True,
    )
    return seq

def build_seq3():
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.Crop(percent=(0, 0.1))),
            iaa.SomeOf(
                (0, 4),
                [
                    iaa.Affine(rotate=(-10, 10), shear=(-6, 6)),
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur((0, 1.5)),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.MedianBlur(k=(3, 7)),
                        ]
                    ),
                    iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.25)),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.03 * 255), per_channel=0.3
                    ),
                    iaa.Add((-5, 5), per_channel=0.3),
                    iaa.Multiply((0.8, 1.2), per_channel=0.3),
                    iaa.LinearContrast((0.8, 1.5), per_channel=0.3),
                    iaa.Grayscale(alpha=(0.0, 0.1)),
                ],
                random_order=True,
            ),
        ],
        random_order=True,
    )
    return seq
class ImageBatchProcessor:
    def __init__(
        self,
        image_folder,
        label_folder,
        target_image_folder,
        target_label_folder,
        batch_size=16,
    ):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.target_image_folder = target_image_folder
        self.target_label_folder = target_label_folder
        self.batch_size = batch_size
        self.aug = build_seq3()
        self._reset_output_folders()

    def _reset_output_folders(self):
        for folder in [self.target_image_folder, self.target_label_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

    def augment_and_save(self, nb_augmentation=3):
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

        time_start = time.time()
        print(f"Starting augmentation of {len(image_files)} images...")
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(self._load_single_image_and_labels, image_files)
            )
        print(f"Loaded {len(results)} images and labels successfully.")
        with ProcessPoolExecutor() as executor:
            print(f"Augmenting images with {nb_augmentation} augmentations each...")
            augmented_image_count = 0
            for i, result in enumerate(results):
                if result[0] is not None:  # Check if image is loaded
                    augmented_image_count += 1
                    executor.submit(self._augment_and_save_single_image, (i, result, nb_augmentation))

        time_end = time.time()
        print(f"Total augmentation time: {time_end - time_start:.2f} seconds")
        print(f"Total images augmented: {augmented_image_count}")

    def _load_single_image_and_labels(self, image_file):
        img_path = os.path.join(self.image_folder, image_file)
        img = cv2.imread(img_path)
        if img is not None:
            bbs = self._load_bounding_boxes(image_file, img.shape)
            return img, bbs
        else:
            print(f"Warning: {img_path} could not be read.")
            return None, None

    def _load_bounding_boxes(self, image_file, shape):
        label_file = os.path.splitext(image_file)[0] + ".csv"
        label_path = os.path.join(self.label_folder, label_file)
        bbs = []
        if os.path.exists(label_path):
            try:
                df = pd.read_csv(label_path)
                if df.empty:
                    #print(f"No bounding boxes found for {image_file}.")
                    return None
                labels = df.to_numpy()
                for bb in labels:
                    bbs.append(
                        BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3], label=bb[4])
                    )
            except pd.errors.EmptyDataError:
                #print(f"No columns to parse from file for {image_file}.")
                return None
        return (
            BoundingBoxesOnImage(bbs, shape).remove_out_of_image().clip_out_of_image()
        )

    def _augment_and_save_single_image(self, args):
        img_idx, result, nb_augmentation = args
        img, bbs = result

        if img is None:
            print(f"Skipping augmentation for image {img_idx} due to loading issues.")
            return

        images = [img] * nb_augmentation
        if bbs is None:
            bbs_batch = [None] * nb_augmentation
        else:
            bbs_batch = [bbs] * nb_augmentation

        batches = [UnnormalizedBatch(images=images, bounding_boxes=bbs_batch)]
        batches_aug = list(self.aug.augment_batches(batches, background=True))

        self._save_augmented_images_and_labels(batches_aug, img_idx, bbs is None)

    def _save_augmented_images_and_labels(self, batches_aug, img_idx, is_none_label):
        for aug_idx, batch in enumerate(batches_aug):
            for img_aug_idx, (img, bbs) in enumerate(
                zip(batch.images_aug, batch.bounding_boxes_aug)
            ):
                img_name = f"aug_{img_idx}_{aug_idx}_{img_aug_idx}.jpg"
                img_path = os.path.join(self.target_image_folder, img_name)
                cv2.imwrite(img_path, img)

                if is_none_label:
                    print(f"we saved image {img_name} without label")
                    continue

                if bbs is None:
                    print(f"we saved image {img_name} without label")
                    continue

                label_name = f"aug_{img_idx}_{aug_idx}_{img_aug_idx}.csv"
                label_path = os.path.join(self.target_label_folder, label_name)
                self._save_bounding_boxes(bbs, label_path)

    def _save_bounding_boxes(self, bbs, label_path):
        data = [
            {"x1": bb.x1, "y1": bb.y1, "x2": bb.x2, "y2": bb.y2, "label": bb.label}
            for bb in bbs
        ]
        df = pd.DataFrame(data)
        df.to_csv(label_path, header=False, index=False)



# Usage example:
# processor = ImageBatchProcessor("path/to/images", "path/to/labels", "path/to/target_images", "path/to/target_labels")
# processor.augment_and_save(nb_augmentation=3)

import os
import random
import shutil
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np

crop_sizes = [
    (50, 50),
    (50, 100),
    (100, 100),
    (100, 200),
    (200, 200),
    (200, 400),
    (400, 400),
    (150, 100),
]


def random_crop_img(img, n_crops=1):
    w, h = img.size
    imgs = []
    for w_crop, h_crop in crop_sizes:
        # random number  20%
        w_crop = int(w_crop + w_crop * random.randint(-20, 20) / 100)
        h_crop = int(h_crop + h_crop * random.randint(-20, 20) / 100)
        # Randomly crop the image n_crops times
        if w_crop > w or h_crop > h:
            continue
        for i in range(n_crops):
            x = np.random.randint(0, w - w_crop)
            y = np.random.randint(0, h - h_crop)
            crop = img.crop((x, y, x + w_crop, y + h_crop))
            imgs.append(crop)
    return imgs


def crop_images_from_folder(image_folder, csv_folder, output_folder):
    # Ensure the output folder exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Iterate through each file in the image folder
    for image_filename in tqdm(
        os.listdir(image_folder), desc=f"Cropping images in {image_folder}"
    ):
        # Get the file name and extension
        name, ext = os.path.splitext(image_filename)

        # Construct the corresponding CSV file path
        csv_filename = f"{name}.csv"
        csv_path = os.path.join(csv_folder, csv_filename)

        # Construct the full image file path
        image_path = os.path.join(image_folder, image_filename)

        # Check if the CSV file exists and is not empty
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            #print(f"CSV does not exist or is empty: {csv_filename}")
            with Image.open(image_path) as img:
                crops = random_crop_img(img)
                for i, crop in enumerate(crops):
                    save_name = f"{name}_cropped_none_{i}{ext}"
                    crop.save(os.path.join(output_folder, save_name))
            continue

        with open(csv_path, "r") as f:
            if f.read().isspace():
                #print(f"CSV file contains only whitespace: {csv_filename}")
                with Image.open(image_path) as img:
                    crops = random_crop_img(img)
                    for i, crop in enumerate(crops):
                        save_name = f"{name}_cropped_none_{i}{ext}"
                        crop.save(os.path.join(output_folder, save_name))
                continue

        # Read the CSV file
        df = pd.read_csv(csv_path, header=None)

        # Open the image
        with Image.open(image_path) as img:
            img_width, img_height = img.size

            # Iterate through each row in the CSV
            for index, row in df.iterrows():
                xmin, ymin, xmax, ymax, label = row[0], row[1], row[2], row[3], row[4]
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                if label != "ff":
                    cropped_image_name = f"{name}_cropped_{label}_{index}{ext}"
                    cropped_image_path = os.path.join(output_folder, cropped_image_name)
                    cropped_img.save(cropped_image_path)

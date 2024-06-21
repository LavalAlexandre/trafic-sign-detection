import os
import csv
import random
import numpy as np
from PIL import Image

def generate_negative_samples(images_dir, labels_dir, negative_samples_dir, target_sizes=[(64, 64), (128, 128)], num_samples=2):
    if not os.path.exists(negative_samples_dir):
        os.makedirs(negative_samples_dir)
    
    sample_count = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.csv'):
            image_path = os.path.join(images_dir, label_file.replace('.csv', '.jpg'))
            label_path = os.path.join(labels_dir, label_file)

            if os.path.exists(image_path):
                image = Image.open(image_path)
                img_width, img_height = image.size

                # Load positive bounding boxes
                bounding_boxes = []
                with open(label_path, mode='r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if row:
                            xmin, ymin, xmax, ymax = map(int, row[:4])
                            bounding_boxes.append((xmin, ymin, xmax, ymax))

                # Generate negative samples
                for _ in range(num_samples):
                    attempts = 0
                    while attempts < 100:  # Avoid infinite loop, limit attempts to 100
                        target_size = random.choice(target_sizes)
                        if img_width > target_size[0] and img_height > target_size[1]:
                            x = np.random.randint(0, img_width - target_size[0])
                            y = np.random.randint(0, img_height - target_size[1])
                            valid = True

                            # Check if the crop overlaps any positive bounding box
                            for xmin, ymin, xmax, ymax in bounding_boxes:
                                if x < xmax and x + target_size[0] > xmin and y < ymax and y + target_size[1] > ymin:
                                    valid = False
                                    break

                            if valid:
                                cropped_image = image.crop((x, y, x + target_size[0], y + target_size[1]))
                                cropped_image = cropped_image.resize(target_size)
                                sample_count += 1
                                cropped_image.save(os.path.join(negative_samples_dir, f"neg_{sample_count}_{target_size[0]}x{target_size[1]}.jpg"))
                                break
                        attempts += 1

    print(f"Generated {sample_count} negative samples.")

# Directories
images_dir_train = "train/images"
labels_dir_train = "train/labels"
negative_samples_dir = "negative_samples"

# Generate and save negative samples
generate_negative_samples(images_dir_train, labels_dir_train, negative_samples_dir, target_sizes=[(64, 64), (128, 128)], num_samples=2)

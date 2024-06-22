import os
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def generate_negative_samples(images_dir, labels_dir, negative_samples_dir, target_sizes=[(64, 64), (128, 128)], num_samples=2,iou_threshold=0.4):
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
                            for xmin, ymin, xmax, ymax in bounding_boxes:
                                crop_box = (x, y, x + target_size[0], y + target_size[1])
                                bounding_box = (xmin, ymin, xmax, ymax)
                                if iou(crop_box, bounding_box) > iou_threshold:
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

def iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # The area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou


def get_negative_prediction(detections, validations, image_folder_path, negative_samples_dir):
    if not os.path.exists(negative_samples_dir):
        os.makedirs(negative_samples_dir)

    negative_predictions = []
    for _, detection_row in detections.iterrows():
        image_id_d, x_min_d, y_min_d, x_max_d, y_max_d, score_d, label_d = detection_row
        no_intersection = True

        for _, validation_row in validations.iterrows():
            image_id_v, x_min_v, y_min_v, x_max_v, y_max_v, label_v = validation_row

            if image_id_v == image_id_d:
                iou_value = iou((x_min_v, y_min_v, x_max_v, y_max_v), (x_min_d, y_min_d, x_max_d, y_max_d))
                if iou_value >= 0.3 or label_v == label_d:
                    no_intersection = False
                    break

        if no_intersection and label_d != 'none':
            negative_predictions.append((image_id_d, x_min_d, y_min_d, x_max_d, y_max_d, label_d))

    for i, (image_id, x_min, y_min, x_max, y_max, label) in enumerate(negative_predictions):
        image_path = os.path.join(image_folder_path, f"{image_id}")
        image = Image.open(image_path)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        save_path = os.path.join(negative_samples_dir, f"negative_sample_{i}_none.png")
        cropped_image.save(save_path)

"""
# Directories
images_dir_train = "train/images"
labels_dir_train = "train/labels"
negative_samples_dir = "negative_samples"

# Generate and save negative samples
generate_negative_samples(images_dir_train, labels_dir_train, negative_samples_dir, target_sizes=[(64, 64), (128, 128)], num_samples=2)
"""
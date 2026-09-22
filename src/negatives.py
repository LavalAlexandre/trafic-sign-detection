"""
Negative ('none') examples for training.

Generated files follow the crop naming scheme `<image>_cropped_none_<suffix>.<ext>`
so both src.dataset and src.cnn read them as 'none'.
"""
import os
import random

from PIL import Image

from src.evaluation import iou, load_ground_truth


def generate_negative_samples(images_dir, labels_dir, negative_samples_dir, target_sizes=((64, 64), (128, 128)),
                              num_samples=2, iou_threshold=0.4, seed=42):
    """Random crops of the train images that don't overlap any annotated box."""
    os.makedirs(negative_samples_dir, exist_ok=True)
    rng = random.Random(seed)
    ground_truth = load_ground_truth(labels_dir)
    boxes_by_image = {image: group[["x1", "y1", "x2", "y2"]].to_numpy()
                      for image, group in ground_truth.groupby("image")}

    sample_count = 0
    for image_name in sorted(os.listdir(images_dir)):
        if not image_name.endswith(".jpg"):
            continue
        image = Image.open(os.path.join(images_dir, image_name))
        img_width, img_height = image.size
        stem = os.path.splitext(image_name)[0]

        for _ in range(num_samples):
            for _attempt in range(100):
                w, h = rng.choice(target_sizes)
                if img_width <= w or img_height <= h:
                    continue
                x = rng.randrange(0, img_width - w)
                y = rng.randrange(0, img_height - h)
                crop_box = (x, y, x + w, y + h)
                if all(iou(crop_box, box) <= iou_threshold for box in boxes_by_image.get(image_name, [])):
                    image.crop(crop_box).save(
                        os.path.join(negative_samples_dir, f"{stem}_cropped_none_rand-{x}-{y}-{w}x{h}.jpg"))
                    sample_count += 1
                    break

    print(f"Generated {sample_count} negative samples.")


def get_negative_prediction(detections, ground_truth, image_folder_path, negative_samples_dir, iou_threshold=0.3):
    """
    Hard negative mining: save every detection that overlaps no ground truth box as a 'none' crop.
    Only run this on training images, never on validation or test images.
    """
    os.makedirs(negative_samples_dir, exist_ok=True)
    boxes_by_image = {image: group[["x1", "y1", "x2", "y2"]].to_numpy(float)
                      for image, group in ground_truth.groupby("image")}

    def overlaps_ground_truth(det):
        det_box = det[["x1", "y1", "x2", "y2"]].to_numpy(float)
        return any(iou(det_box, box) >= iou_threshold for box in boxes_by_image.get(det["image"], []))

    false_positives = detections[~detections.apply(overlaps_ground_truth, axis=1)] if len(detections) else detections

    for _, det in false_positives.iterrows():
        x1, y1, x2, y2 = (int(det[c]) for c in ("x1", "y1", "x2", "y2"))
        stem = os.path.splitext(det["image"])[0]
        with Image.open(os.path.join(image_folder_path, det["image"])) as image:
            image.crop((x1, y1, x2, y2)).save(
                os.path.join(negative_samples_dir, f"{stem}_cropped_none_fp-{x1}-{y1}-{x2}-{y2}.png"))

    print(f"Saved {len(false_positives)} hard negatives to {negative_samples_dir}")
    return false_positives

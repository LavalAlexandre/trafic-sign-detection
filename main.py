"""Train the SVM classifier end to end (augmentation, cropping, training). See machine_learning.ipynb for detection."""
from src.augment import ImageBatchProcessor
from src.crop import crop_images_from_folder
from src.train import train


if __name__ == "__main__":
    image_folder = "dataset/train/images/"
    label_folder = "dataset/train/labels/"
    target_image_folder = "dataset/train/images_aug/"
    target_label_folder = "dataset/train/labels_aug/"
    val_image_folder = "dataset/val/images/"
    val_label_folder = "dataset/val/labels/"

    train_path = "dataset/train/cropped_images/"
    aug_train_path = "dataset/train/aug_cropped_images/"
    val_path = "dataset/val/cropped_images/"

    processor = ImageBatchProcessor(
        image_folder, label_folder, target_image_folder, target_label_folder
    )
    processor.augment_and_save(nb_augmentation=5)

    crop_images_from_folder(image_folder, label_folder, train_path)
    crop_images_from_folder(target_image_folder, target_label_folder, aug_train_path)
    crop_images_from_folder(val_image_folder, val_label_folder, val_path)

    train(
        train_path=train_path,
        aug_train_path=aug_train_path,
        val_path=val_path,
        seed=42,
        label_size_factor=1,
        standard_size=(64, 64),
        max_iter=10000,
    )

from src.train import train
from src.augment import ImageBatchProcessor
from src.crop import crop_images_from_folder


if __name__ == "__main__":
    # data_augment()

    image_folder = "dataset/train/images/"
    label_folder = "dataset/train/labels/"
    target_image_folder = "dataset/train/images_aug/"
    target_label_folder = "dataset/train/labels_aug/"
    val_path = "dataset/val/images/"
    val_label_path = "dataset/val/labels/"

    # """
    processor = ImageBatchProcessor(
        image_folder, label_folder, target_image_folder, target_label_folder
    )
    processor.augment_and_save(nb_augmentation=5)
    # """
    # """
    crop_images_from_folder(
        image_folder, label_folder, "dataset/train/cropped_images/"
    )
    crop_images_from_folder(
        target_image_folder, target_label_folder, "dataset/train/aug_cropped_images/"
    )
    crop_images_from_folder(val_path, val_label_path, "dataset/val/cropped_images/")
    # """
    train_path = "dataset/train/cropped_images/"
    val_path = "dataset/val/cropped_images/"
    aug_train_path = "dataset/train/aug_cropped_images/"



    standard_size = (32, 32)

    train(
        train_path=train_path,
        aug_train_path=aug_train_path,
        val_path=val_path,
        seed=42,
        label_size_factor=0,
        standard_size=standard_size,
        max_iter=1000000,
    )

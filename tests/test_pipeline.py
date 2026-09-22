"""End-to-end smoke tests of both pipelines on a small synthetic dataset."""
import os

import pandas as pd

from src.augment import ImageBatchProcessor
from src.crop import crop_images_from_folder
from src.dataset import dataset
from src.detection import detection_images_in_folder
from src.evaluation import evaluate_detections, load_detections, load_ground_truth
from src.model import model
from src.negatives import generate_negative_samples, get_negative_prediction


def test_augmentation_keeps_every_box(synthetic_dataset, tmp_path):
    image_dir, label_dir = synthetic_dataset["train"]
    ImageBatchProcessor(str(image_dir), str(label_dir), str(tmp_path / "img"), str(tmp_path / "lbl"),
                        ).augment_and_save(nb_augmentation=1)
    # every train image is annotated, so every augmented image must have a label file
    assert len(os.listdir(tmp_path / "lbl")) == len(os.listdir(image_dir))
    n_boxes = sum(len(pd.read_csv(tmp_path / "lbl" / f, header=None)) for f in os.listdir(tmp_path / "lbl"))
    assert n_boxes > len(os.listdir(image_dir))  # half the images have two boxes


def test_svm_pipeline(synthetic_dataset, tmp_path):
    train_images, train_labels = synthetic_dataset["train"]
    val_images, val_labels = synthetic_dataset["val"]
    train_crops, val_crops = tmp_path / "train_crops", tmp_path / "val_crops"
    crop_images_from_folder(str(train_images), str(train_labels), str(train_crops))
    crop_images_from_folder(str(val_images), str(val_labels), str(val_crops))
    generate_negative_samples(str(train_images), str(train_labels), str(train_crops),
                              target_sizes=[(64, 64), (100, 100)], num_samples=4)

    train_data = dataset(str(train_crops) + "/", label_size_factor=0, undersample={})
    val_data = dataset(str(val_crops) + "/", train=False)
    assert {im.label for im in train_data.images} == {"interdiction", "fvert", "none"}
    assert {im.label for im in val_data.images} == {"interdiction", "fvert"}

    clf = model(seed=0)
    clf.train_svm(train_data, max_iter=10000)
    assert clf.evaluate(val_data) > 0.9

    detections_csv = tmp_path / "detections.csv"
    detection_images_in_folder(str(train_images), clf, str(detections_csv), output_folder=None, n_jobs=2,
                               min_region_size=1000, sign_threshold=0.5, light_threshold=0.5)
    detections = load_detections(detections_csv)
    assert len(detections) > 0

    get_negative_prediction(detections, load_ground_truth(str(train_labels)), str(train_images), str(train_crops))

    detection_images_in_folder(str(val_images), clf, str(detections_csv), output_folder=str(tmp_path / "out"),
                               n_jobs=1, min_region_size=1000, sign_threshold=0.5, light_threshold=0.5)
    report = evaluate_detections(load_detections(detections_csv), load_ground_truth(str(val_labels)), plot=False)
    assert set(report.index) <= {"interdiction", "fvert"}
    assert report["recall"].max() > 0


def test_cnn_pipeline(synthetic_dataset, tmp_path):
    from src.cnn import Sign_Classifier, TrafficSignDataset

    train_images, train_labels = synthetic_dataset["train"]
    val_images, val_labels = synthetic_dataset["val"]
    train_crops, val_crops = tmp_path / "train_crops", tmp_path / "val_crops"
    crop_images_from_folder(str(train_images), str(train_labels), str(train_crops))
    crop_images_from_folder(str(val_images), str(val_labels), str(val_crops))
    generate_negative_samples(str(train_images), str(train_labels), str(train_crops), num_samples=2)

    weights = tmp_path / "cnn.pth"
    clf = Sign_Classifier(device="cpu").fit(TrafficSignDataset(str(train_crops)), TrafficSignDataset(str(val_crops)),
                                            n_epoch=2, batch_size=8, save_path=str(weights))
    clf = Sign_Classifier(device="cpu").load_model(str(weights))

    detection_images_in_folder(str(val_images), clf, str(tmp_path / "detections.csv"), output_folder=None,
                               n_jobs=1, detector="sliding_window", step_size_ratio=0.5,
                               sign_threshold=0.0, light_threshold=0.0)
    detections = load_detections(tmp_path / "detections.csv")
    assert set(detections["label"]) <= {"danger", "interdiction", "ceder", "obligation", "frouge", "fvert",
                                        "forange", "stop"}

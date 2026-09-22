import numpy as np
import pandas as pd
import pytest

from src.detection import non_max_suppression, selective_search
from src.evaluation import evaluate_detections, iou, load_ground_truth, match_detections
from src.labels import label_from_filename


@pytest.mark.parametrize("filename, label", [
    ("0106_cropped_fvert_1.jpg", "fvert"),
    ("aug_3_0_1_cropped_ceder_0.jpg", "ceder"),
    ("0042_cropped_none_fp-1-2-3-4.png", "none"),
    ("neg_12_64x64.jpg", "none"),
    ("0001_cropped_stop.jpg", "stop"),
])
def test_label_from_filename(filename, label):
    assert label_from_filename(filename) == label


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_nms_keeps_best_and_drops_nested_boxes():
    detections = [
        (0, 0, 100, 100, "stop", 0.9),
        (5, 5, 105, 105, "stop", 0.95),
        (20, 20, 40, 40, "danger", 0.99),  # highest score -> kept, and suppresses nothing else
        (300, 300, 350, 350, "ceder", 0.5),
    ]
    picked = non_max_suppression(detections, 0.1)
    assert [d[5] for d in picked] == [0.99, 0.95, 0.5]
    picked = non_max_suppression(detections[:2] + detections[3:], 0.1)
    assert [d[5] for d in picked] == [0.95, 0.5]


def test_each_ground_truth_is_matched_once():
    gt = pd.DataFrame([["a.jpg", 0, 0, 100, 100, "stop"]], columns=["image", "x1", "y1", "x2", "y2", "label"])
    det = pd.DataFrame([
        ["a.jpg", 0, 0, 100, 100, 0.9, "stop"],
        ["a.jpg", 2, 2, 100, 100, 0.8, "stop"],
    ], columns=["image", "x1", "y1", "x2", "y2", "score", "label"])
    matches = match_detections(det, gt)
    assert matches.tolist() == [0, None]

    report = evaluate_detections(det, gt, plot=False)
    assert report.loc["stop", "precision"] == 0.5
    assert report.loc["stop", "recall"] == 1.0
    assert report.loc["stop", "AP"] == pytest.approx(1.0)


def test_load_ground_truth_skips_empty_and_ff(tmp_path):
    (tmp_path / "0001.csv").write_text("1,2,3,4,stop\n5,6,7,8,ff\n")
    (tmp_path / "0002.csv").write_text("")
    gt = load_ground_truth(tmp_path)
    assert gt.values.tolist() == [["0001.jpg", 1, 2, 3, 4, "stop"]]


def test_selective_search_sees_the_real_image():
    image = np.zeros((200, 200, 3), np.uint8)
    image[50:150, 50:150] = (0, 0, 255)
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in selective_search(image, min_region_size=100)]
    assert max(iou(box, (50, 50, 150, 150)) for box in boxes) > 0.9

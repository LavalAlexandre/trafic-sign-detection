import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.labels import IGNORED_LABEL

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)


def load_ground_truth(label_dir):
    """One `<image>.csv` per image with `x1,y1,x2,y2,label` rows and no header."""
    dfs = []
    for csv_file in sorted(os.listdir(label_dir)):
        if not csv_file.endswith(".csv"):
            continue
        path = os.path.join(label_dir, csv_file)
        if os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path, header=None, names=BOX_COLUMNS + ["label"])
        except pd.errors.EmptyDataError:
            continue
        df.insert(0, "image", os.path.splitext(csv_file)[0] + ".jpg")
        dfs.append(df)

    gt = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["image"] + BOX_COLUMNS + ["label"])
    return gt[gt["label"] != IGNORED_LABEL].reset_index(drop=True)


def load_detections(csv_path):
    """Read the csv written by src.detection.detection_images_in_folder."""
    df = pd.read_csv(csv_path)
    df.columns = ["image"] + BOX_COLUMNS + ["score", "label"]
    return df


def match_detections(detections, ground_truth, iou_threshold=0.5, same_label=False):
    """
    Greedily match detections (highest score first) to ground truth boxes of the same image.
    Each ground truth box is matched at most once.
    Returns, for every detection, the index of its ground truth box in `ground_truth` (or None).
    """
    matches = pd.Series([None] * len(detections), index=detections.index, dtype=object)
    gt_by_image = {image: group for image, group in ground_truth.groupby("image")}
    used = set()

    for det_idx, det in detections.sort_values("score", ascending=False).iterrows():
        candidates = gt_by_image.get(det["image"])
        if candidates is None:
            continue
        best_iou, best_idx = iou_threshold, None
        for gt_idx, gt in candidates.iterrows():
            if gt_idx in used or (same_label and gt["label"] != det["label"]):
                continue
            overlap = iou(det[BOX_COLUMNS].to_numpy(float), gt[BOX_COLUMNS].to_numpy(float))
            if overlap >= best_iou:
                best_iou, best_idx = overlap, gt_idx
        if best_idx is not None:
            used.add(best_idx)
            matches[det_idx] = best_idx

    return matches


def average_precision(detections, ground_truth, label, iou_threshold=0.5):
    """VOC-style AP (all-point interpolation) for one class."""
    dets = detections[detections["label"] == label].sort_values("score", ascending=False)
    gts = ground_truth[ground_truth["label"] == label]
    if len(gts) == 0:
        return np.nan
    if len(dets) == 0:
        return 0.0

    tp = match_detections(dets, gts, iou_threshold).loc[dets.index].notna().to_numpy()
    tp_cum = np.cumsum(tp)
    recall = tp_cum / len(gts)
    precision = tp_cum / np.arange(1, len(tp) + 1)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def evaluate_detections(detections, ground_truth, iou_threshold=0.5, plot=True):
    """
    Confusion matrix, per-class precision / recall and mAP of detections against ground truth.
    A detection that matches no ground truth box counts as a false positive (truth = 'none'),
    a ground truth box that matches no detection as a false negative (prediction = 'none').
    """
    matches = match_detections(detections, ground_truth, iou_threshold)

    y_true = [ground_truth.at[m, "label"] if pd.notna(m) else "none" for m in matches]
    y_pred = list(detections["label"])
    missed = ground_truth.index.difference(matches.dropna().astype(int))
    y_true += list(ground_truth.loc[missed, "label"])
    y_pred += ["none"] * len(missed)

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    for i, label in enumerate(labels):
        if label == "none":
            continue
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        rows.append({
            "label": label,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "AP": average_precision(detections, ground_truth, label, iou_threshold),
            "support": int(cm[i, :].sum()),
        })
    report = pd.DataFrame(rows).set_index("label")

    if plot:
        with np.errstate(invalid="ignore"):
            cm_normalized = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt=".0%", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()

    print(report.round(3))
    print(f"\nmAP@{iou_threshold}: {report['AP'].mean():.3f}")
    return report

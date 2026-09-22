"""
Detection pipeline shared by the ML (src.model) and DL (src.cnn) classifiers.

Any classifier works as long as it exposes `predict(window) -> (label, confidence)`,
where `window` is a BGR image crop as returned by cv2.imread.
"""
import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import selectivesearch
from tqdm import tqdm

from src.labels import TRAFFIC_LIGHTS

DETECTION_COLUMNS = ['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe']
# Traffic lights are roughly 2.2 times taller than wide
LIGHT_ASPECT_RATIO = 2.2


def accept(label, confidence, sign_threshold, light_threshold):
    if label == "none":
        return False
    threshold = light_threshold if label in TRAFFIC_LIGHTS else sign_threshold
    return confidence > threshold


# %% sliding window with image pyramid
def pyramid(image, scale=1.1, min_size_ratio=0.1):
    yield image
    min_height = int(image.shape[0] * min_size_ratio)
    min_width = int(image.shape[1] * min_size_ratio)
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, int(w * image.shape[0] / image.shape[1])))
        if image.shape[0] < min_height or image.shape[1] < min_width:
            break
        yield image


def sliding_window(image, step_size_ratio, window_width):
    height, width = image.shape[:2]
    window_height = window_width
    step_size = max(1, int(window_width * step_size_ratio))
    for y in range(0, height - window_height, step_size):
        for x in range(0, width - window_width, step_size):
            yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])


def detect_traffic_signs(image, model, step_size_ratio=0.5, window_size_ratio=0.1, pyramid_scale=1.1,
                         sign_threshold=0.8, light_threshold=0.8):
    detections = []
    window_width = int(image.shape[1] * window_size_ratio)
    for resized in pyramid(image, scale=pyramid_scale, min_size_ratio=0.05):
        for (x, y, (win_width, win_height), window) in sliding_window(resized, step_size_ratio, window_width):
            label, confidence = model.predict(window)
            if accept(label, confidence, sign_threshold, light_threshold):
                x_orig = int(x * (image.shape[1] / resized.shape[1]))
                y_orig = int(y * (image.shape[0] / resized.shape[0]))
                w_orig = int(win_width * (image.shape[1] / resized.shape[1]))
                h_orig = int(win_height * (image.shape[0] / resized.shape[0]))
                detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, label, confidence))
    return detections


# %% multi-scale sliding window without pyramid
def sliding_window_without_piramid(image, step_size_ratio, max_window_size_ratio, min_window_width=150):
    """Square windows for signs, shrinking by 10% of max_window_size_ratio * width at each scale."""
    height, width = image.shape[:2]
    max_window_width = int(width * max_window_size_ratio)
    size_ratio = max_window_size_ratio - 0.1
    window_width = int(max_window_width * size_ratio)

    while window_width >= min_window_width:
        step_size = max(1, int(window_width * step_size_ratio))
        for y in range(0, height - window_width, step_size):
            for x in range(0, width - window_width, step_size):
                yield (x, y, (window_width, window_width), image[y:y + window_width, x:x + window_width])

        size_ratio = size_ratio - 0.1
        window_width = int(max_window_width * size_ratio)


def sliding_window_without_piramid_feu(image, step_size_ratio, max_window_size_ratio, min_window_height=200):
    """Tall windows for traffic lights, shrinking by 10% of max_window_size_ratio * height at each scale."""
    height, width = image.shape[:2]
    max_window_height = int(height * max_window_size_ratio)
    size_ratio = max_window_size_ratio - 0.1
    window_height = int(max_window_height * size_ratio)

    while window_height >= min_window_height:
        window_width = int(window_height / LIGHT_ASPECT_RATIO)
        step_size = max(1, int(window_height * step_size_ratio))
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, max(1, step_size // 2)):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])

        size_ratio = size_ratio - 0.1
        window_height = int(max_window_height * size_ratio)


def detect_traffic_signs_without_piramid(image, model, step_size_ratio=0.2, max_window_size_ratio=1.0,
                                         sign_threshold=0.8, light_threshold=0.8):
    detections = []

    # square windows only keep signs, tall windows only keep traffic lights
    for windows, keep_lights in (
            (sliding_window_without_piramid(image, step_size_ratio, max_window_size_ratio), False),
            (sliding_window_without_piramid_feu(image, step_size_ratio, max_window_size_ratio), True),
    ):
        for (x, y, (win_width, win_height), window) in windows:
            label, confidence = model.predict(window)
            if (label in TRAFFIC_LIGHTS) == keep_lights and accept(label, confidence, sign_threshold, light_threshold):
                detections.append((x, y, x + win_width, y + win_height, label, confidence))

    return detections


# %% selective search
def selective_search(image, min_region_size=2500):
    _, regions = selectivesearch.selective_search(image, scale=250, sigma=0.9, min_size=100)

    candidates = set()
    for r in regions:
        # skip small regions
        if r['size'] < min_region_size:
            continue
        x, y, w, h = r['rect']
        # skip distorted regions
        if w == 0 or h == 0 or w / h > 2 or h / w > 3:
            continue
        candidates.add(r['rect'])

    return candidates


def detect_traffic_signs_with_selective_search(image, model, min_region_size=2500,
                                               sign_threshold=0.8, light_threshold=0.8):
    detections = []
    for (x, y, w, h) in selective_search(image, min_region_size):
        window = image[y:y + h, x:x + w]
        label, confidence = model.predict(window)
        if accept(label, confidence, sign_threshold, light_threshold):
            detections.append((x, y, x + w, y + h, label, confidence))

    return detections


def non_max_suppression(detections, overlap_thresh):
    """
    Greedy NMS over (x1, y1, x2, y2, label, confidence) tuples, highest confidence first.
    A box is suppressed if more than `overlap_thresh` of its own area is covered by a kept box,
    or if it lies entirely inside a kept box.
    """
    if len(detections) == 0:
        return []

    coords = np.array([d[:4] for d in detections], dtype=int)
    x1, y1, x2, y2 = coords.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort([d[5] for d in detections])[::-1]

    picked = []
    while len(idxs) > 0:
        i, rest = idxs[0], idxs[1:]
        picked.append(detections[i])

        w = np.maximum(0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        overlap = (w * h) / areas[rest]
        inside = (x1[rest] >= x1[i]) & (y1[rest] >= y1[i]) & (x2[rest] <= x2[i]) & (y2[rest] <= y2[i])

        idxs = rest[~((overlap > overlap_thresh) | inside)]

    return picked


def draw_detections(image, detections):
    for (x1, y1, x2, y2, label, confidence) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
    return image


# %% detection on a folder
DETECTORS = {
    "selective_search": detect_traffic_signs_with_selective_search,
    "sliding_window": detect_traffic_signs_without_piramid,
    "pyramid": detect_traffic_signs,
}


def process_image(filename, folder_path, model, output_folder, detector, overlap_thresh, detector_kwargs):
    image = cv2.imread(os.path.join(folder_path, filename))
    detections = DETECTORS[detector](image, model, **detector_kwargs)
    picked = non_max_suppression(detections, overlap_thresh)

    if output_folder is not None:
        cv2.imwrite(os.path.join(output_folder, filename), draw_detections(image, picked))

    return [[filename, x1, y1, x2, y2, float(confidence), label]
            for (x1, y1, x2, y2, label, confidence) in picked]


def detection_images_in_folder(folder_path, model, csv_path, detector="selective_search", overlap_thresh=0.1,
                               output_folder="output_images", n_jobs=None, **detector_kwargs):
    """
    Run `detector` on every .jpg of `folder_path` and write the detections to `csv_path`.
    Annotated images go to `output_folder` (None to skip). n_jobs=None uses every CPU;
    use n_jobs=1 for torch models, which already use several threads.
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    filenames = sorted(f for f in os.listdir(folder_path) if f.endswith(".jpg"))
    func = partial(process_image, folder_path=folder_path, model=model, output_folder=output_folder,
                   detector=detector, overlap_thresh=overlap_thresh, detector_kwargs=detector_kwargs)
    progress = partial(tqdm, total=len(filenames), desc="Detecting traffic signs")

    if n_jobs == 1:
        per_image = list(progress(map(func, filenames)))
    else:
        with Pool(processes=n_jobs or os.cpu_count()) as pool:
            per_image = list(progress(pool.imap(func, filenames)))

    df = pd.DataFrame([row for rows in per_image for row in rows], columns=DETECTION_COLUMNS)
    df.to_csv(csv_path, index=False)
    return df


# %% interactive
def select_region_and_predict(image_path, model):
    """Draw a box with the mouse, press 'c' to classify it, 'r' to reset, '0' to quit."""
    ref_point = []
    image = cv2.imread(image_path)
    clone = image.copy()

    def click_and_crop(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point[:] = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            image[:] = clone

        elif key == ord('c') and len(ref_point) == 2:
            (x1, y1), (x2, y2) = ref_point
            label, confidence = model.predict(clone[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)])
            cv2.putText(image, f"{label} ({confidence:.2f})", (min(x1, x2), min(y1, y2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ref_point.clear()

        elif key == ord('0'):
            break

    cv2.destroyAllWindows()

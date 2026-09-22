import os
import sys

import cv2
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Synthetic stand-ins for real classes: a red disc ('interdiction') and a green bar ('fvert')
SHAPES = {
    "interdiction": lambda img, x1, y1, x2, y2: cv2.circle(
        img, ((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1) // 2, (0, 0, 255), -1),
    "fvert": lambda img, x1, y1, x2, y2: cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), -1),
}


def make_split(root, n_images, seed):
    """Images with one or two objects each, plus one-per-image `x1,y1,x2,y2,label` csv files."""
    rng = np.random.default_rng(seed)
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for i in range(n_images):
        img = np.full((240, 320, 3), 90, np.uint8)
        img += rng.integers(0, 40, img.shape, dtype=np.uint8)
        rows = []
        for j, label in enumerate(rng.permutation(list(SHAPES))[: 1 + i % 2]):
            size = int(rng.integers(50, 70))
            x1, y1 = 20 + 150 * j + int(rng.integers(0, 40)), int(rng.integers(10, 240 - size - 10))
            x2, y2 = x1 + size, y1 + (size if label == "interdiction" else int(size * 2))
            y2 = min(y2, 235)
            SHAPES[label](img, x1, y1, x2, y2)
            rows.append(f"{x1},{y1},{x2},{y2},{label}")
        cv2.imwrite(str(image_dir / f"{i:04d}.jpg"), img)
        (label_dir / f"{i:04d}.csv").write_text("\n".join(rows) + "\n")
    return image_dir, label_dir


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    return {
        "train": make_split(root / "train", 12, seed=0),
        "val": make_split(root / "val", 6, seed=1),
        "root": root,
    }

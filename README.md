# Road Sign and Traffic Light Detection and Classification
This project detects and recognizes road signs and traffic lights with two approaches: traditional machine learning and deep learning. Both use the same region proposals (selective search or multi-scale sliding windows) and non-maximum suppression. They differ only in the classifier:

- **Machine learning**: HOG, HSV color moments and a 4×4 brightness grid, classified by a linear SVM (`src/model.py`)
- **Deep learning**: a ResNet18-shaped CNN on 64×64 crops (`src/cnn.py`)

Both pipelines use hard negative mining: after each training round, detections on the train images that overlap no annotated box are added to the training set as `none`.

Classes: `danger`, `interdiction`, `ceder`, `obligation`, `stop`, `frouge`, `forange`, `fvert` (red / orange / green traffic lights) and `none`.

## Setup
Requires [uv](https://docs.astral.sh/uv/).
```bash
uv sync
```
`model.train_xgboost` needs the optional extra: `uv sync --extra xgboost`. On macOS it also needs `brew install libomp`.

## Dataset
The dataset isn't included. Put it at the repository root:
```
dataset/
├── train/{images,labels}/
├── val/{images,labels}/
└── test/images/
```
Each `labels/<image>.csv` has one `x1,y1,x2,y2,label` row per object and no header. Boxes labelled `ff` are ignored.

## Run
Open one of the notebooks from the repository root (`uv run jupyter lab`, or select `.venv` as the kernel in your editor):
- `machine_learning.ipynb`
- `deep_learning.ipynb`

Both notebooks train on `train`, report per-class precision / recall and mAP@0.5 on `val`, then run detection on `test`. `main.py` only trains the SVM.

## Code layout
| Module | Role |
| --- | --- |
| `src/augment.py` | imgaug data augmentation of the train images and their boxes |
| `src/crop.py` | crops annotated boxes (plus random background crops) into `<image>_cropped_<label>_<i>.jpg` |
| `src/negatives.py` | random negative crops and hard negative mining |
| `src/image.py`, `src/dataset.py`, `src/model.py` | ML features, dataset and classifiers |
| `src/cnn.py` | DL dataset and classifier |
| `src/detection.py` | region proposals, NMS, detection on a folder |
| `src/evaluation.py` | ground truth loading, detection matching, precision / recall / mAP |

## Models
Trained SVMs from the three hard-negative-mining rounds are stored in `src/models` (64×64 features). Load one with `model().load("src/models/third_training.pkl")`. They were trained before the fixes to validation leakage, augmentation and negative mining, so retraining will give different results.

## Tests
```bash
uv run pytest
```
The tests run both pipelines end to end on a small synthetic dataset.

## Author
Laval Alexandre

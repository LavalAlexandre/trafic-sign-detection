import os

LABELS = [
    "danger",
    "interdiction",
    "ceder",
    "obligation",
    "frouge",
    "fvert",
    "forange",
    "stop",
    "none",
]
TRAFFIC_LIGHTS = {"frouge", "forange", "fvert"}
# "ff" boxes are annotated but excluded from training and evaluation
IGNORED_LABEL = "ff"


def label_from_filename(filename):
    """Crops are named `<image>_cropped_<label>_<suffix>.<ext>`; anything else is a negative."""
    parts = os.path.splitext(filename)[0].split("_")
    try:
        label = parts[parts.index("cropped") + 1]
    except (ValueError, IndexError):
        return "none"
    return label if label in LABELS else "none"

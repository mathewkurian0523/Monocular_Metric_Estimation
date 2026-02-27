import numpy as np
import os

try:
    DEFAULT_REF_MIN_CONF = float(os.getenv("SCENESCALE_REF_MIN_CONF", "0.6"))
except ValueError:
    DEFAULT_REF_MIN_CONF = 0.6


# -----------------------------
# Basic Bounding Box Utilities
# -----------------------------

def bbox_height(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, y2 - y1)


def bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1)


def bbox_area(bbox):
    return bbox_width(bbox) * bbox_height(bbox)


# -----------------------------
# Reference Filtering
# -----------------------------

def is_valid_reference(obj, reference_db, min_confidence=None):
    """
    Checks if detection qualifies as a reference object.
    """

    cls = obj["class"]
    confidence = obj["confidence"]
    pixel_size = obj["pixel_size"]

    if cls not in reference_db:
        return False

    if min_confidence is None:
        min_confidence = DEFAULT_REF_MIN_CONF

    if confidence < min_confidence:
        return False

    if pixel_size < reference_db[cls]["min_pixel_size"]:
        return False

    return True


# -----------------------------
# Robust Scale Filtering
# -----------------------------

def filter_outliers(scales, tolerance=0.2):
    """
    Removes scale values that deviate more than
    tolerance (20%) from the median.
    """

    if len(scales) == 0:
        return []

    median = np.median(scales)

    filtered = [
        s for s in scales
        if abs(s - median) / median < tolerance
    ]

    return filtered


def compute_scale_stability(scales):
    """
    Returns standard deviation and coefficient of variation.
    """

    if len(scales) < 2:
        return 0.0, 0.0

    std = np.std(scales)
    mean = np.mean(scales)

    if mean == 0:
        return std, 0.0

    coeff_var = std / mean

    return std, coeff_var

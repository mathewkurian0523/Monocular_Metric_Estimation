# geometry/scale_estimator.py

import numpy as np
from geometry.utils import (
    is_valid_reference,
    filter_outliers,
    compute_scale_stability
)


def estimate_scale(reference_objects, reference_db):
    """
    reference_objects format:
    [
        {
            "class": str,
            "pixel_size": float,
            "confidence": float
        }
    ]
    """

    valid_scales = []
    weights = []

    # -----------------------------
    # 1️⃣ Validate and compute per-object scale
    # -----------------------------
    for obj in reference_objects:

        if not is_valid_reference(obj, reference_db):
            continue

        cls = obj["class"]
        pixel_size = obj["pixel_size"]

        real_mean = reference_db[cls]["mean_m"]
        reliability = reference_db[cls]["reliability"]

        if pixel_size <= 0:
            continue

        scale = real_mean / pixel_size

        # Weight combines detection confidence and class reliability
        weight = obj["confidence"] * reliability

        valid_scales.append(scale)
        weights.append(weight)

    if len(valid_scales) == 0:
        return {
            "status": "no_valid_references"
        }

    # -----------------------------
    # 2️⃣ Median-based outlier rejection
    # -----------------------------
    filtered_scales = filter_outliers(valid_scales)

    if len(filtered_scales) == 0:
        return {
            "status": "all_references_filtered"
        }

    # Keep corresponding weights
    filtered_weights = [
        weights[i]
        for i, s in enumerate(valid_scales)
        if s in filtered_scales
    ]

    # -----------------------------
    # 3️⃣ Weighted aggregation
    # -----------------------------
    estimated_scale = np.average(filtered_scales, weights=filtered_weights)

    # -----------------------------
    # 4️⃣ Stability check
    # -----------------------------
    scale_std, coeff_var = compute_scale_stability(filtered_scales)

    # If scale variation too high → unstable
    if coeff_var > 0.25:
        return {
            "status": "unstable_scale",
            "estimated_scale": float(estimated_scale),
            "scale_std": float(scale_std),
            "num_references": len(filtered_scales),
            "confidence": float(np.mean(filtered_weights))
        }

    # -----------------------------
    # 5️⃣ Return stable scale
    # -----------------------------
    return {
        "status": "stable",
        "estimated_scale": float(estimated_scale),
        "scale_std": float(scale_std),
        "num_references": len(filtered_scales),
        "confidence": float(np.mean(filtered_weights))
    }
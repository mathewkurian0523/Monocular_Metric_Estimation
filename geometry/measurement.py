# geometry/measurement.py

import numpy as np


def measure_pixel_dimension(pixel_value, scale_output, sigma_pixel=2.0):
    """
    pixel_value: float (width or height in pixels)
    scale_output: dict from estimate_scale
    sigma_pixel: assumed pixel measurement noise (default ±2 pixels)
    """

    if scale_output is None:
        return {"status": "no_scale_available"}

    if scale_output.get("status") != "stable":
        return {"status": "unstable_scale"}

    scale = scale_output["estimated_scale"]
    scale_std = scale_output["scale_std"]

    # Real-world measurement
    real_value = pixel_value * scale

    # Uncertainty propagation
    variance_scale = (pixel_value * scale_std) ** 2
    variance_pixel = (scale * sigma_pixel) ** 2

    total_uncertainty = np.sqrt(variance_scale + variance_pixel)

    return {
        "status": "measured",
        "real_value_m": float(real_value),
        "uncertainty_m": float(total_uncertainty),
        "confidence": scale_output["confidence"],
        "num_references": scale_output["num_references"]
    }
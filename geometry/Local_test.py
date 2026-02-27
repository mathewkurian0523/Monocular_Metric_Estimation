from geometry.reference_db import REFERENCE_DB
from geometry.scale_estimator import estimate_scale
from geometry.measurement import measure_pixel_dimension

reference_objects = [
    {"class": "chair", "pixel_size": 300, "confidence": 0.9},
    {"class": "chair", "pixel_size": 320, "confidence": 0.8},
]

scale = estimate_scale(reference_objects, REFERENCE_DB)

print("Scale:", scale)

measurement = measure_pixel_dimension(1000, scale)

print("Measurement:", measurement)
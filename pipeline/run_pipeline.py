# pipeline/run_pipeline.py

import cv2

from detection.detector import detect_objects
from geometry.reference_db import REFERENCE_DB
from geometry.utils import bbox_height, bbox_width
from geometry.scale_estimator import estimate_scale
from geometry.measurement import measure_pixel_dimension
from visualization.draw_boxes import draw_boxes


def run_pipeline(image_path, target_class):
    """
    image_path: path to input image
    target_class: class you want to measure
    """

    # -----------------------------
    # 1️⃣ Load Image
    # -----------------------------
    image = cv2.imread(image_path)

    if image is None:
        return {"status": "image_load_failed"}

    # -----------------------------
    # 2️⃣ Run Object Detection
    # -----------------------------
    detections = detect(image)
    # Expected format:
    # [
    #   {
    #     "class": str,
    #     "bbox": [x1, y1, x2, y2],
    #     "confidence": float
    #   }
    # ]

    if not detections:
        return {"status": "no_detections"}

    # -----------------------------
    # 3️⃣ Split Reference & Target
    # -----------------------------
    reference_objects = []
    target_objects = []

    for det in detections:
        cls = det["class"]
        bbox = det["bbox"]

        # Compute pixel height (we use vertical dimension for scale)
        pixel_size = bbox_height(bbox)

        obj_data = {
            "class": cls,
            "pixel_size": pixel_size,
            "confidence": det["confidence"]
        }

        if cls in REFERENCE_DB:
            reference_objects.append(obj_data)

        if cls == target_class:
            target_objects.append(det)

    if len(reference_objects) == 0:
        return {"status": "no_reference_objects"}

    if len(target_objects) == 0:
        return {"status": "target_not_found"}

    # -----------------------------
    # 4️⃣ Estimate Scale
    # -----------------------------
    scale_output = estimate_scale(reference_objects, REFERENCE_DB)

    if scale_output.get("status") != "stable":
        return scale_output  # pass instability reason upward

    # -----------------------------
    # 5️⃣ Measure Target Object
    # -----------------------------
    measurements = []

    for target in target_objects:
        bbox = target["bbox"]

        # You can choose height or width depending on use case
        pixel_dimension = bbox_width(bbox)

        measurement = measure_pixel_dimension(
            pixel_dimension,
            scale_output
        )

        measurements.append({
            "bbox": bbox,
            "measurement": measurement
        })

    # -----------------------------
    # 6️⃣ Visualization
    # -----------------------------
    annotated_image = draw_boxes(image.copy(), detections)

    # -----------------------------
    # 7️⃣ Final Output
    # -----------------------------
    return {
        "status": "success",
        "scale": scale_output,
        "measurements": measurements,
        "annotated_image": annotated_image
    }
# tests/test_pipeline_full_eval.py

import os
import cv2
import pandas as pd
import numpy as np

from detection.detector import detect
from geometry.reference_db import REFERENCE_DB
from geometry.utils import bbox_height, bbox_width
from geometry.scale_estimator import estimate_scale
from geometry.measurement import measure_pixel_dimension


IMAGE_DIR = "data/test_images"
CSV_PATH = "data/test_images/metric_ground_truth.csv"


def percentage_error(pred, gt):
    return abs(pred - gt) / gt * 100


def main():

    df = pd.read_csv(CSV_PATH)

    total_width_errors = []
    total_height_errors = []

    for image_name in os.listdir(IMAGE_DIR):

        if not image_name.lower().endswith((".jpg", ".png")):
            continue

        frame_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(IMAGE_DIR, image_name)

        print(f"\nProcessing: {image_name}")

        image = cv2.imread(image_path)
        detections = detect(image)

        if not detections:
            print("No detections.")
            continue

        # -----------------------------
        # Build Reference Objects
        # -----------------------------
        reference_objects = []

        for det in detections:
            cls = det["class"]
            bbox = det["bbox"]

            pixel_size = bbox_height(bbox)

            if cls in REFERENCE_DB:
                reference_objects.append({
                    "class": cls,
                    "pixel_size": pixel_size,
                    "confidence": det["confidence"]
                })

        if not reference_objects:
            print("No valid references.")
            continue

        scale_output = estimate_scale(reference_objects, REFERENCE_DB)

        if scale_output.get("status") != "stable":
            print("Scale unstable.")
            continue

        # -----------------------------
        # Evaluate Each Detection
        # -----------------------------
        gt_rows = df[df["Frame"] == frame_name]

        for det in detections:

            cls = det["class"]
            bbox = det["bbox"]

            # Match with GT
            gt_match = gt_rows[gt_rows["ObjectName"] == cls]

            if gt_match.empty:
                continue

            gt_width = float(gt_match.iloc[0]["RealWidth_m"])
            gt_height = float(gt_match.iloc[0]["RealHeight_m"])

            pixel_w = bbox_width(bbox)
            pixel_h = bbox_height(bbox)

            # Measure width
            width_measurement = measure_pixel_dimension(pixel_w, scale_output)
            height_measurement = measure_pixel_dimension(pixel_h, scale_output)

            if width_measurement["status"] != "measured":
                continue

            pred_width = width_measurement["real_value_m"]
            pred_height = height_measurement["real_value_m"]

            width_error = percentage_error(pred_width, gt_width)
            height_error = percentage_error(pred_height, gt_height)

            total_width_errors.append(width_error)
            total_height_errors.append(height_error)

            print(f"Object: {cls}")
            print(f"  Width Error: {width_error:.2f}%")
            print(f"  Height Error: {height_error:.2f}%")

    # -----------------------------
    # Final Summary
    # -----------------------------
    if total_width_errors:

        print("\n==============================")
        print("FINAL SUMMARY")
        print("==============================")

        print(f"Objects Evaluated: {len(total_width_errors)}")
        print(f"Average Width Error: {np.mean(total_width_errors):.2f}%")
        print(f"Average Height Error: {np.mean(total_height_errors):.2f}%")

        print(f"Worst Width Error: {np.max(total_width_errors):.2f}%")
        print(f"Worst Height Error: {np.max(total_height_errors):.2f}%")

    else:
        print("\nNo valid evaluations performed.")


if __name__ == "__main__":
    main()
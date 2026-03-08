# tools/debug_visualize.py

import cv2
import os

from detection.detector import detect_objects
from geometry.reference_db import REFERENCE_DB
from geometry.scale_estimator import estimate_scale
from geometry.measurement import measure_pixel_dimension


IMAGE_PATH = "data/test_images/tatas.jpeg"
DEBUG = os.getenv("SCENESCALE_DEBUG", "0") == "1"


def draw_text(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA
    )


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Image not found.")
        return

    detections = detect_objects(IMAGE_PATH)

    if not detections:
        print("No detections.")
        return
    if DEBUG:
        class_counts = {}
        for det in detections:
            class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1
        print(f"[debug] detections_by_class={class_counts}")

    # -----------------------------
    # Build reference objects
    # -----------------------------
    reference_objects = []

    for det in detections:
        if det["is_reference"] and det["class"] in REFERENCE_DB:
            ref_dimension = REFERENCE_DB[det["class"]].get("dimension", "height")
            if ref_dimension == "width":
                pixel_size = det["width_px"]
            else:
                pixel_size = det["height_px"]

            if DEBUG:
                print(
                    f"[debug] ref class={det['class']} conf={det['confidence']:.3f} "
                    f"pixel_size={pixel_size} req_dim={ref_dimension} "
                    f"min_pixel={REFERENCE_DB[det['class']]['min_pixel_size']}"
                )

            reference_objects.append({
                "class": det["class"],
                "pixel_size": pixel_size,
                "confidence": det["confidence"]
            })

    if not reference_objects:
        print("No reference objects found.")
        return

    scale_output = estimate_scale(reference_objects, REFERENCE_DB)

    print("\n===== SCALE OUTPUT =====")
    print(scale_output)

    if scale_output.get("status") != "stable":
        print("Scale unstable.")
        return

    scale_value = scale_output["estimated_scale"]

    # -----------------------------
    # Draw detections + measurements
    # -----------------------------
    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]

        pixel_w = det["width_px"]
        pixel_h = det["height_px"]

        width_measurement = measure_pixel_dimension(pixel_w, scale_output)
        height_measurement = measure_pixel_dimension(pixel_h, scale_output)

        if width_measurement["status"] != "measured":
            continue

        real_w = width_measurement["real_value_m"]
        real_h = height_measurement["real_value_m"]

        # Draw bounding box
        color = (0, 255, 0) if det["is_reference"] else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{cls}"
        dim_text = f"W:{real_w:.2f}m H:{real_h:.2f}m"

        draw_text(image, label, x1, y1 - 20, color)
        draw_text(image, dim_text, x1, y1 - 5, color)

    # Draw scale info at top
    scale_text = f"Scale: {scale_value:.6f} m/px"
    draw_text(image, scale_text, 20, 30, (0, 255, 255))

    # -----------------------------
    # Show Image
    # -----------------------------
    cv2.imshow("SceneScale Debug", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

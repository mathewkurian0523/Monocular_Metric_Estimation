from ultralytics import YOLO
import os

# Load model once
MODEL_NAME = "yolov8n.pt"
model = YOLO(MODEL_NAME)

CONF_THRESHOLD = 0.35


def detect_objects(image_path: str) -> list:
    """
    Detect objects in an image and return standardized detections
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(image_path)[0]

    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 3)
        })

    return detections
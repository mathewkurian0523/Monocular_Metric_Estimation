import os
import threading

from ultralytics import YOLO

from geometry.reference_db import REFERENCE_DB

_model_lock = threading.Lock()

# ---------------- CONFIG ---------------- #

MODEL_NAME = os.getenv("SCENESCALE_MODEL", "yolov8n.pt")
try:
    CONF_THRESHOLD = float(os.getenv("SCENESCALE_CONF", "0.35"))
except ValueError:
    CONF_THRESHOLD = 0.35
DEBUG = os.getenv("SCENESCALE_DEBUG", "0") == "1"

# Normalize classes from COCO/custom models to pipeline keys.
CLASS_ALIASES = {
    # COCO names
    "cell_phone": "phone",
    "remote": "phone",
    "book": "books",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "dining_table": "table",
    # Custom detector names
    "a4_paper": "a4_sheet",
    "a4_sheet": "a4_sheet",
    "debit_card": "credit_card",
    "credit_card": "credit_card",
    "coin": "coin",
    "books": "books",
    "table": "table",
    "suitcase": "suitcase",
    "phone": "phone",
}

_model = None


def _normalize_class_name(name: str):
    key = name.strip().lower().replace(" ", "_")
    return CLASS_ALIASES.get(key)


def _load_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_NAME)
                _model.overrides["verbose"] = False


def detect_objects(image_path: str) -> list:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    _load_model()

    try:
        results = _model(image_path)[0]
    except Exception as e:
        raise RuntimeError(f"Detection failed: {str(e)}")

    if DEBUG:
        print(f"[detector] model={MODEL_NAME} conf={CONF_THRESHOLD} image={image_path}")
        print(f"[detector] raw_boxes={len(results.boxes)}")

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            if DEBUG:
                cls_id = int(box.cls[0])
                print(f"[detector] skip(low_conf) class={_model.names[cls_id]} conf={conf:.3f}")
            continue

        cls_id = int(box.cls[0])
        class_name = _model.names[cls_id]
        project_class = _normalize_class_name(class_name)
        if project_class is None:
            if DEBUG:
                print(f"[detector] skip(unmapped_class) class={class_name}")
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width_px = x2 - x1
        height_px = y2 - y1

        if height_px >= width_px:
            ref_axis = "height"
            ref_pixels = height_px
        else:
            ref_axis = "width"
            ref_pixels = width_px

        detections.append(
            {
                "class": project_class,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "width_px": width_px,
                "height_px": height_px,
                "reference_pixels": ref_pixels,
                "reference_axis": ref_axis,
                "is_reference": project_class in REFERENCE_DB,
            }
        )
        if DEBUG:
            print(f"[detector] keep class={project_class} conf={conf:.3f}")

    if DEBUG:
        print(f"[detector] final_detections={len(detections)}")
    return detections

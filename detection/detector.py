

from ultralytics import YOLO
import os
import threading
_model_lock = threading.Lock()

# ---------------- CONFIG ---------------- #

MODEL_NAME = "yolov8n.pt"   # downloads automatically first run
CONF_THRESHOLD = 0.35

# COCO → Project classes
CLASS_MAP = {
    "cell phone": "phone",
    "remote": "phone",
    "book": "a4_paper",
    "laptop": "laptop",
    "keyboard": "keyboard",

    # Ignore unstable scale references
    "person": None,
    "chair": None,
    "cup": None,
    "bottle": None,
    "tv": None,
    "dining table": None
}

# Define which classes can act as scale references
CLASS_INFO = {
    "phone": {"is_reference": True},
    "keyboard": {"is_reference": True},
    "a4_paper": {"is_reference": True},
    "laptop": {"is_reference": False}
}

# ------------- MODEL LOADING ------------ #

_model = None

def _load_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_NAME)
                _model.overrides['verbose'] = False

# ------------- MAIN API FUNCTION -------- #

def detect_objects(image_path: str) -> list:
    """
    Detect objects and return standardized detections.

    Returns:
    [
        {
            "class": "phone",
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.91,
            "width_px": int,
            "height_px": int,
            "reference_pixels": int,
            "reference_axis": "width" | "height",
            "is_reference": bool
        }
    ]
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    _load_model()

    try:
          results = _model(image_path)[0]
    except Exception as e:
          raise RuntimeError(f"Detection failed: {str(e)}")
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        class_name = _model.names[cls_id]

        # Normalize class
        project_class = CLASS_MAP.get(class_name, None)
        if project_class is None:
            continue

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width_px = x2 - x1
        height_px = y2 - y1

        # Determine orientation-aware reference axis
        if height_px >= width_px:
            ref_axis = "height"
            ref_pixels = height_px
        else:
            ref_axis = "width"
            ref_pixels = width_px

        detections.append({
            "class": project_class,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 3),
            "width_px": width_px,
            "height_px": height_px,
            "reference_pixels": ref_pixels,
            "reference_axis": ref_axis,
            "is_reference": CLASS_INFO.get(project_class, {}).get("is_reference", False)
        })

    return detections
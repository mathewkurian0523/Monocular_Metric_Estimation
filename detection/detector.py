import os
import threading
from pathlib import Path

from ultralytics import YOLO

from geometry.reference_db import REFERENCE_DB

_model_lock = threading.Lock()

# ---------------- CONFIG ---------------- #

SINGLE_MODEL_NAME = os.getenv("SCENESCALE_MODEL", "yolov8n.pt")
REFERENCE_MODEL_NAME = os.getenv("SCENESCALE_REFERENCE_MODEL")
TARGET_MODEL_NAME = os.getenv("SCENESCALE_TARGET_MODEL")
DISABLE_DUAL = os.getenv("SCENESCALE_DISABLE_DUAL", "0") == "1"

_project_root = Path(__file__).resolve().parents[1]
_default_reference_model = _project_root / "models" / "reference_detector.pt"
_default_target_model = _project_root / "models" / "target_detector.pt"

if not REFERENCE_MODEL_NAME and _default_reference_model.exists():
    REFERENCE_MODEL_NAME = str(_default_reference_model)
if not TARGET_MODEL_NAME and _default_target_model.exists():
    TARGET_MODEL_NAME = str(_default_target_model)

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

# Classes that may exist in REFERENCE_DB but should be treated as measurement targets.
NON_REFERENCE_CLASSES = {"laptop"}

_single_model = None
_reference_model = None
_target_model = None


def _normalize_class_name(name: str):
    key = name.strip().lower().replace(" ", "_")
    return CLASS_ALIASES.get(key)


def _normalize_raw_key(name: str):
    return name.strip().lower().replace(" ", "_")


def _load_single_model():
    global _single_model
    if _single_model is None:
        with _model_lock:
            if _single_model is None:
                _single_model = YOLO(SINGLE_MODEL_NAME)
                _single_model.overrides["verbose"] = False


def _load_dual_models():
    global _reference_model, _target_model
    if _reference_model is None or _target_model is None:
        with _model_lock:
            if _reference_model is None:
                _reference_model = YOLO(REFERENCE_MODEL_NAME)
                _reference_model.overrides["verbose"] = False
            if _target_model is None:
                _target_model = YOLO(TARGET_MODEL_NAME)
                _target_model.overrides["verbose"] = False


def _parse_results(model, results, source):
    detections = []
    if DEBUG:
        print(f"[detector] source={source} raw_boxes={len(results.boxes)}")

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if conf < CONF_THRESHOLD:
            if DEBUG:
                print(f"[detector] source={source} skip(low_conf) class={class_name} conf={conf:.3f}")
            continue

        project_class = _normalize_class_name(class_name)
        if project_class is None:
            # Keep target/single detections even when alias mapping is missing.
            # This prevents valid target classes from being dropped due to label mismatch.
            if source in ("target", "single"):
                project_class = _normalize_raw_key(class_name)
                if DEBUG:
                    print(
                        f"[detector] source={source} keep(unmapped_as_raw) "
                        f"class={class_name} -> {project_class}"
                    )
            else:
                if DEBUG:
                    print(f"[detector] source={source} skip(unmapped_class) class={class_name}")
                continue

        is_reference = project_class in REFERENCE_DB and project_class not in NON_REFERENCE_CLASSES

        # Keep references from reference model; keep targets from target model.
        if source == "reference" and not is_reference:
            if DEBUG:
                print(
                    f"[detector] source={source} skip(non_reference_in_reference_model) "
                    f"class={project_class} conf={conf:.3f}"
                )
            continue
        if source == "target" and is_reference:
            if DEBUG:
                print(
                    f"[detector] source={source} skip(reference_in_target_model) "
                    f"class={project_class} conf={conf:.3f}"
                )
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
                "is_reference": is_reference,
            }
        )
        if DEBUG:
            print(f"[detector] source={source} keep class={project_class} conf={conf:.3f}")

    return detections


def detect_objects(image_path: str) -> list:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    use_dual = bool(not DISABLE_DUAL and REFERENCE_MODEL_NAME and TARGET_MODEL_NAME)
    if DEBUG:
        print(
            f"[detector] resolved_models single={SINGLE_MODEL_NAME} "
            f"reference={REFERENCE_MODEL_NAME} target={TARGET_MODEL_NAME}"
        )

    if use_dual:
        _load_dual_models()
        if DEBUG:
            print(
                f"[detector] mode=dual ref_model={REFERENCE_MODEL_NAME} "
                f"target_model={TARGET_MODEL_NAME} conf={CONF_THRESHOLD} image={image_path}"
            )

        try:
          
            ref_results = _reference_model(str(image_path))[0]
            tgt_results = _target_model(image_path)[0]
        except Exception as e:
            raise RuntimeError(f"Detection failed: {str(e)}")

        detections = []
        detections.extend(_parse_results(_reference_model, ref_results, "reference"))
        detections.extend(_parse_results(_target_model, tgt_results, "target"))
    else:
        _load_single_model()
        if DEBUG:
            if DISABLE_DUAL:
                print("[detector] dual mode disabled via SCENESCALE_DISABLE_DUAL=1")
            print(
                f"[detector] mode=single model={SINGLE_MODEL_NAME} "
                f"conf={CONF_THRESHOLD} image={image_path}"
            )

        try:
            results = _single_model(image_path)[0]
        except Exception as e:
            raise RuntimeError(f"Detection failed: {str(e)}")

        detections = _parse_results(_single_model, results, "single")

    if DEBUG:
        ref_count = sum(1 for d in detections if d["is_reference"])
        tgt_count = len(detections) - ref_count
        print(
            f"[detector] final_detections={len(detections)} "
            f"references={ref_count} targets={tgt_count}"
        )

    return detections

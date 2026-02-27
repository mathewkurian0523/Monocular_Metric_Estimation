import cv2
import json
from ultralytics import YOLO

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IMAGE_PATH = "sample_test_images/IMG_5789.jpg"

REFERENCE_MODEL_PATH = "models/reference_detector.pt"
TARGET_MODEL_PATH    = "models/target_detector.pt"

CONF_THRESHOLD = 0.25

REFERENCE_CLASSES = {"a4_paper", "coin", "debit_card"}

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading models...")
ref_model = YOLO(REFERENCE_MODEL_PATH)
tgt_model = YOLO(TARGET_MODEL_PATH)

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------

def run_model(model, image, obj_type):
    results = model(image, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(float, box.xyxy[0])

        detections.append({
            "class": cls_name,
            "type": obj_type,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return detections


# --------------------------------------------------
# MAIN
# --------------------------------------------------

print("Running inference...")

img = cv2.imread(IMAGE_PATH)

ref_detections = run_model(ref_model, IMAGE_PATH, "reference")
tgt_detections = run_model(tgt_model, IMAGE_PATH, "target")

# Remove reference objects accidentally detected by target model
filtered_targets = [
    d for d in tgt_detections if d["class"] not in REFERENCE_CLASSES
]

final_detections = ref_detections + filtered_targets

# --------------------------------------------------
# PRINT JSON OUTPUT (SCALER INPUT)
# --------------------------------------------------

print("\nFINAL OUTPUT FOR SCALER:\n")
print(json.dumps(final_detections, indent=2))


# --------------------------------------------------
# OPTIONAL: DRAW RESULTS
# --------------------------------------------------

for det in final_detections:
    x1, y1, x2, y2 = map(int, det["bbox"])

    color = (0,255,0) if det["type"]=="reference" else (255,0,0)
    label = f"{det['class']} {det['confidence']:.2f}"

    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
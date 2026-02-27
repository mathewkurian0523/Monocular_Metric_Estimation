from ultralytics import YOLO

# Load models once (fast)
ref_model = YOLO("models/reference_detector.pt")
target_model = YOLO("models/target_detector.pt")

REFERENCE_CLASSES = {"a4_paper", "coin", "debit_card"}


def parse_results(results):
    detections = []
    r = results[0]

    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": xyxy
        })

    return detections


def run_inference(image_path):

    # 1️⃣ run both models
    ref_results = ref_model(image_path)
    target_results = target_model(image_path)

    ref_dets = parse_results(ref_results)
    target_dets = parse_results(target_results)

    # 2️⃣ remove reference objects from target detections
    filtered_target = [
        d for d in target_dets if d["class"] not in REFERENCE_CLASSES
    ]

    # 3️⃣ merge (reference detections override target)
    final_detections = ref_dets + filtered_target

    return final_detections


if __name__ == "__main__":
    image = "sample_test_images/testtest.jpg"
    detections = run_inference(image)

    import json
print(json.dumps(detections, indent=2))
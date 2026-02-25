from detection.detector import detect_objects

detections = detect_objects("image.png")

for d in detections:
    print(d)
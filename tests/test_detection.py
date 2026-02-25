from detection.detector import detect_objects

if __name__ == "__main__":
    detections = detect_objects("image.jpg")

    for d in detections:
        print(d)
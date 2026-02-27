from ultralytics import YOLO
from pathlib import Path

# absolute project root (folder containing this file -> SceneScale)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "datasets" / "reference_objects" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATASET_PATH),
        epochs=80,
        imgsz=640,
        batch=8,
        patience=20,
        project=str(RUNS_DIR),      # << forces correct save folder
        name="reference_detector"
    )

if __name__ == "__main__":
    train()
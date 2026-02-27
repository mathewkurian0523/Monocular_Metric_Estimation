import csv

from geometry.measurement import measure_pixel_dimension
from geometry.reference_db import REFERENCE_DB
from geometry.scale_estimator import estimate_scale
from geometry.utils import bbox_height, bbox_width

CSV_PATH = "data/test_images/ground_truth.csv"


def percentage_error(pred, gt):
    return abs(pred - gt) / gt * 100


def normalize_label(label: str) -> str:
    """Map dataset labels to reference-db keys used in the current codebase."""
    key = label.strip().lower().replace(" ", "_")
    aliases = {
        "a4_sheet": "a4_sheet",
        "credit_card": "credit_card",
        "coin": "coin",
        "laptop": "laptop",
        "table": "table",
        "books": "books",
        "suitcase": "suitcase",
    }
    return aliases.get(key, key)


def test_ground_truth_schema_matches_pipeline_expectations():
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {
            "Frame",
            "ObjectID",
            "ObjectName",
            "RealWidth_m",
            "RealHeight_m",
            "Distance_m",
        }
        assert expected_columns.issubset(set(reader.fieldnames or []))


def test_scale_and_measurement_from_reference_objects():
    # Simulate detections in the format used by geometry modules.
    reference_objects = [
        {"class": "credit_card", "pixel_size": 100, "confidence": 0.95},
        {"class": "coin", "pixel_size": 30, "confidence": 0.90},
        {"class": "a4_sheet", "pixel_size": 350, "confidence": 0.92},
    ]

    scale_output = estimate_scale(reference_objects, REFERENCE_DB)

    assert scale_output["status"] == "stable"
    assert scale_output["estimated_scale"] > 0

    pixel_w = bbox_width([10, 20, 210, 220])
    pixel_h = bbox_height([10, 20, 210, 220])

    width_measurement = measure_pixel_dimension(pixel_w, scale_output)
    height_measurement = measure_pixel_dimension(pixel_h, scale_output)

    assert width_measurement["status"] == "measured"
    assert height_measurement["status"] == "measured"
    assert width_measurement["real_value_m"] > 0
    assert height_measurement["real_value_m"] > 0


def test_label_normalization_and_error_formula():
    assert normalize_label("A4 sheet") == "a4_sheet"
    assert normalize_label("Credit card") == "credit_card"
    assert abs(percentage_error(1.2, 1.0) - 20.0) < 1e-9
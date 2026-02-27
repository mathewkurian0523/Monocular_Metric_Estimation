# geometry/reference_db.py

REFERENCE_DB = {
    "a4_sheet": {
        "dimension": "height",
        "mean_m": 0.297,          # Standard A4 height
        "std_m": 0.001,           # Highly standardized
        "reliability": 0.99,
        "min_pixel_size": 50 
    },
    "credit_card": {
        "dimension": "width",
        "mean_m": 0.085,          # Standard ID-1 width
        "std_m": 0.0005,
        "reliability": 0.98,
        "min_pixel_size": 30
    },
    "coin": {
        "dimension": "height",    # Diameter in this context
        "mean_m": 0.025,          # 2.5cm diameter
        "std_m": 0.002,
        "reliability": 0.95,
        "min_pixel_size": 20
    },
    "laptop": {
        "dimension": "width",
        "mean_m": 0.35,           # Approx. 15-inch laptop width
        "std_m": 0.03,
        "reliability": 0.75,
        "min_pixel_size": 100
    },
    "table": {
        "dimension": "height",
        "mean_m": 0.75,           # Matches your previous table height
        "std_m": 0.05,
        "reliability": 0.85,
        "min_pixel_size": 150
    },
    "suitcase": {
        "dimension": "height",
        "mean_m": 0.42,           # Based on your Blue Case bounds
        "std_m": 0.05,
        "reliability": 0.7,
        "min_pixel_size": 120
    },
    "books": {
        "dimension": "height",
        "mean_m": 0.3,            # Based on your cube scale
        "std_m": 0.05,
        "reliability": 0.6,
        "min_pixel_size": 80
    }
}
"""
Centralized configuration for the Deepfake Detection project.
All paths are relative to the project root directory.
"""

import os
import cv2

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Model Paths
# ============================================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector.h5")

# Use OpenCV's built-in Haar Cascade (works on any system)
HAAR_CASCADE_PATH = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
)

# ============================================================
# Directory Paths
# ============================================================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "output")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# Analysis Thresholds & Parameters
# ============================================================
BIAS_SHIFT = 0.2855                 # Model bias correction
COMBINED_THRESHOLD = 0.5            # Forensic combined score threshold
DEEPFAKE_THRESHOLD = 0.5            # Final classification threshold
MAX_UPLOAD_SIZE_MB = 50             # Maximum upload file size in MB
DEFAULT_FRAME_NUMBER = 12           # Default frame to analyze
IMG_SIZE = (300, 300)               # Model input image size
BATCH_SIZE = 32                     # Training batch size
EPOCHS = 10                        # Training epochs

# ============================================================
# Flask Configuration
# ============================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

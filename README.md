# Video-Based Deepfake Detector

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey.svg)

An advanced AI-powered web application for detecting deepfakes in video files. This project combines deep learning classification with forensic analysis techniques to provide a comprehensive deepfake detection system.

## 🌟 Features

- **Deep Learning Classification**: Utilizes a fine-tuned EfficientNetB0 model to classify faces extracted from video frames.
- **Forensic Analysis**:
  - **Compression Artifacts**: Analyzes Discrete Cosine Transform (DCT) heatmaps to detect double-compression common in deepfakes.
  - **Noise Residuals**: Extracts and scores image noise to find inconsistencies.
  - **Metadata Anomalies**: Inspects video metadata (using FFprobe) for suspicious encoders, odd frame rates, or inconsistent resolutions.
- **Explainable AI (Grad-CAM)**: Generates heatmaps to highlight the regions of the face that influenced the model's decision.
- **Modern Web Interface**: A sleek, responsive UI with a dark/light theme, glitch animations, and particle effects.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) (Ensure `ffprobe` is in your system's PATH)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Deepfake-Detection.git
   cd Deepfake-Detection
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or train the model**:
   - Place your trained model (`deepfake_detector.h5`) in the `models/` directory.
   - If you need to train a new model, see the Training section below.

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Upload a video file, select a frame number to analyze, and view the results!

## 🧠 Model Training & Data Prep

If you want to train the model from scratch on your own dataset:

1. **Prepare Data**: Place your real and fake face images in `processed_data/faces` and `processed_data/fake_faces`.
2. **Split Dataset**:
   ```bash
   python scripts/dataset_split.py
   ```
   This script will filter blurry/invalid faces and split the data into `train`, `val`, and `test` sets inside `dataset_split/`.
3. **Train Model**:
   ```bash
   python scripts/train_model.py
   ```
   The trained model will be saved to `models/deepfake_effnet.h5`.

## 📁 Project Structure

```
Deepfake-Detection/
├── app.py                   # Main Flask application
├── config.py                # Centralized configuration and paths
├── requirements.txt         # Python dependencies
├── scripts/                 # Core logic and utilities
│   ├── analyze_video.py     # Deepfake analysis pipeline
│   ├── train_model.py       # Model training script
│   ├── dataset_split.py     # Dataset processing utility
│   └── preprocess.py        # Additional preprocessing tools
├── models/                  # Directory for trained models
├── templates/               # HTML templates (index.html)
└── static/                  # Static assets and generated outputs
    ├── uploads/             # Temporarily stored uploaded videos
    ├── output/              # Generated analysis outputs
    ├── frames/              # Extracted video frames
    └── heatmaps/            # Generated Grad-CAM heatmaps
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

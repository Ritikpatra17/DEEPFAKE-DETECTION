import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import subprocess
import json
import logging
from datetime import datetime

# Add project root to path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_PATH, HAAR_CASCADE_PATH, BIAS_SHIFT, COMBINED_THRESHOLD, DEEPFAKE_THRESHOLD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variables for model and face cascade (will be initialized later)
model = None
face_cascade = None

# GPU Setup
def setup_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logging.info(f"Running on GPU: {gpus[0]}")
            print(f"✅ Running on GPU: {gpus[0]}")
        else:
            logging.warning("No GPU found, running on CPU")
            print("❌ No GPU found, running on CPU")
    except RuntimeError as e:
        logging.error(f"GPU setup failed: {e}")
        print(f"❌ GPU setup failed: {e}")

# Initialize model and face cascade
def initialize_model_and_cascade():
    global model, face_cascade
    if model is None or face_cascade is None:
        setup_gpu()
        try:
            model = load_model(MODEL_PATH)
            logging.info(f"Loaded model from {MODEL_PATH}")
            print(f"✅ Loaded model from {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            print(f"❌ Failed to load model from {MODEL_PATH}: {e}")
            raise
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            logging.error(f"Failed to load Haar Cascade from {HAAR_CASCADE_PATH}")
            print(f"❌ Failed to load Haar Cascade from {HAAR_CASCADE_PATH}")
            raise ValueError(f"Failed to load Haar Cascade from {HAAR_CASCADE_PATH}")

# Extract frame from video
def extract_frame(video_path, frame_number, output_dir, target_size=(300, 300)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        print(f"❌ Failed to open video file: {video_path}")
        raise ValueError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        logging.error(f"Frame number {frame_number} is out of range (0 to {total_frames-1})")
        print(f"❌ Frame number {frame_number} is out of range (0 to {total_frames-1})")
        raise ValueError(f"Frame number {frame_number} is out of range (0 to {total_frames-1})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        logging.error(f"Failed to extract frame {frame_number} from video: {video_path}")
        print(f"❌ Failed to extract frame {frame_number} from video: {video_path}")
        raise ValueError(f"Failed to extract frame {frame_number} from video: {video_path}")

    frame = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_path = os.path.join(output_dir, f"frame_{frame_number}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    cap.release()
    logging.info(f"Extracted frame {frame_number} saved to {frame_path}")
    print(f"✅ Extracted frame {frame_number}")
    print(f"✅ Saved extracted frame to {frame_path}")
    return frame_rgb, frame_path

# Compression Artifact Analysis with DCT Heatmap
def analyze_compression_artifacts(frame, block_size=8, output_path=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    ph = int(np.ceil(h / block_size) * block_size)
    pw = int(np.ceil(w / block_size) * block_size)
    pad = np.zeros((ph, pw), gray.dtype)
    pad[:h, :w] = gray
    scores = np.zeros((ph // block_size, pw // block_size))

    for y in range(0, ph, block_size):
        for x in range(0, pw, block_size):
            block = pad[y:y+block_size, x:x+block_size]
            dct = cv2.dct(block)
            hf = dct.copy()
            hf[0,0] = 0
            scores[y // block_size, x // block_size] = np.mean(np.abs(hf))

    compression_score = float(np.mean(scores))
    logging.info(f"Compression-artifact score: {compression_score:.3f}")
    print(f"Compression-artifact score: {compression_score:.3f}")
    print("ℹ️ Note: Higher compression scores may indicate double-compression, common in deepfakes. Compare with a known real video.")

    if output_path:
        heatmap = (scores / scores.max() * 255).astype(np.uint8)
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, heatmap)
        logging.info(f"Compression artifact heatmap saved to {output_path}")
        print(f"✅ Saved compression heatmap to {output_path}")

    return compression_score

# Noise Residual Analysis
def analyze_noise_residuals(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residuals = gray - blurred
    noise_score = np.std(residuals)
    logging.info(f"Noise score: {noise_score:.3f}")
    print(f"Noise score: {noise_score:.3f}")
    return noise_score

# Check if ffprobe is available
def check_ffprobe():
    try:
        subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logging.info("FFmpeg (ffprobe) is available.")
        print("✅ FFmpeg (ffprobe) is available.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("ffprobe not found. Skipping metadata anomaly detection.")
        print("⚠️ ffprobe not found. Ensure FFmpeg is installed and added to your PATH. Skipping metadata anomaly detection.")
        return False

# Metadata Anomaly Detection
def extract_metadata(video_path):
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        logging.error(f"ffprobe error: {p.stderr.decode()}")
        print(f"❌ ffprobe error: {p.stderr.decode()}")
        raise RuntimeError(f"ffprobe error: {p.stderr.decode()}")
    return json.loads(p.stdout)

def detect_metadata_anomalies(video_path, expected_frame_rates=(23.976, 24.0, 25.0, 29.97, 30.0)):
    if not check_ffprobe():
        return {"skipped": True, "reason": "ffprobe not found"}

    md = extract_metadata(video_path)
    flags = {}

    video_stream = next((s for s in md['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        logging.error("No video stream found in metadata")
        print("❌ No video stream found in metadata")
        raise ValueError("No video stream found in metadata")

    vr = video_stream.get('r_frame_rate', '0/1').split('/')
    frame_rate = float(vr[0]) / float(vr[1]) if len(vr) == 2 and float(vr[1]) != 0 else 0
    flags['odd_frame_rate'] = bool(not any(abs(frame_rate - f) < 0.01 for f in expected_frame_rates))

    flags['missing_creation_time'] = bool('creation_time' not in md['format'].get('tags', {}))

    sizes = {(s.get('width'), s.get('height')) for s in md['streams'] if s['codec_type'] == 'video'}
    flags['inconsistent_resolution'] = bool(len(sizes) > 1)

    encoder = md['format'].get('tags', {}).get('encoder', '').lower()
    suspicious_encoders = ['deepfake', 'faceswap', 'dfaker']
    flags['suspicious_encoder'] = bool(any(e in encoder for e in suspicious_encoders))

    logging.info("Metadata anomalies: " + str(flags))
    print("Metadata anomalies:")
    for k, v in flags.items():
        print(f"  {k}: {v}")
    return flags

# Detect and crop face
def detect_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        logging.warning("No face detected, using the entire frame")
        print("⚠️ No face detected, using the entire frame")
        return img, (0, 0, img.shape[1], img.shape[0])
    
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    padding_x = int(w * 0.2)
    padding_y = int(h * 0.2)
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(img.shape[1] - x, w + 2 * padding_x)
    h = min(img.shape[0] - y, h + 2 * padding_y)
    
    face_img = img[y:y+h, x:x+w]
    logging.info(f"Face detected and cropped at (x={x}, y={y}, w={w}, h={h})")
    print(f"✅ Face detected and cropped at (x={x}, y={y}, w={w}, h={h})")
    return face_img, (x, y, w, h)

# Load and preprocess image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Failed to load image from {image_path}")
        print(f"❌ Failed to load image from {image_path}")
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_img, face_coords = detect_and_crop_face(img)
    x, y, w, h = face_coords
    
    face_resized = cv2.resize(face_img, (300, 300))
    img_array = face_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    logging.info("Image loaded and preprocessed")
    print("✅ Image loaded and preprocessed")
    return img, face_img, img_array, (x, y, w, h)

# Grad-CAM implementation
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_mean(tf.multiply(output, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heatmap = np.max(heatmap)
    if max_heatmap > 0:
        heatmap = heatmap / (max_heatmap + 1e-10)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

# Overlay heatmap
def overlay_heatmap(heatmap, face_img, original_img, face_coords):
    x, y, w, h = face_coords
    
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.clip(heatmap * 2 - 0.5, 0, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    face_img_rgb = face_img if face_img.shape[-1] == 3 else cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    superimposed_face = heatmap_colored * 0.4 + face_img_rgb * 0.6
    superimposed_face = np.clip(superimposed_face, 0, 255).astype(np.uint8)
    
    result_img = original_img.copy()
    result_img[y:y+h, x:x+w] = superimposed_face
    return result_img

# Adjust probability to reduce model bias
def adjust_model_bias(probability):
    shifted_prob = max(0.0, min(1.0, probability - BIAS_SHIFT))
    logging.info(f"Adjusting model bias: {probability:.4f} → {shifted_prob:.4f}")
    return shifted_prob

# Adjust probability based on compression and noise scores
def adjust_probability_with_metrics(probability, compression_score, noise_score):
    norm_compression = compression_score / 7.0
    norm_noise = noise_score / 10.0
    combined_score = 0.5 * norm_compression + 0.5 * norm_noise
    if combined_score < COMBINED_THRESHOLD:
        adjusted_prob = min(probability * 0.49 / 0.5, 0.49)
        logging.info(f"Combined score ({combined_score:.3f}) < {COMBINED_THRESHOLD}, adjusting probability from {probability:.4f} to {adjusted_prob:.4f}")
    else:
        adjusted_prob = min(max(0.51, probability * 0.99 / 0.5), 0.99)
        logging.info(f"Combined score ({combined_score:.3f}) > {COMBINED_THRESHOLD}, adjusting probability from {probability:.4f} to {adjusted_prob:.4f}")
    return adjusted_prob

# Final classification with metadata anomalies
def final_classification(probability, meta_flags):
    adjusted_prob = probability
    if meta_flags.get('suspicious_encoder', False):
        adjusted_prob = min(0.99, adjusted_prob + 0.05)
        logging.info(f"Suspicious encoder detected, increasing probability: {probability:.4f} → {adjusted_prob:.4f}")
        print(f"ℹ️ Suspicious encoder detected, adjusted probability: {probability:.4f} → {adjusted_prob:.4f}")
    return adjusted_prob

# Main analysis function
def analyze_video(video_path, frame_number, output_dir):
    # Clear TensorFlow session to ensure consistent predictions
    tf.keras.backend.clear_session()

    # Initialize model and face cascade if not already done
    initialize_model_and_cascade()

    # Validate video path
    if not os.path.exists(video_path):
        if not os.path.splitext(video_path)[1]:
            video_path_with_ext = video_path + ".mp4"
            if os.path.exists(video_path_with_ext):
                video_path = video_path_with_ext
                logging.info(f"Video path updated to: {video_path}")
                print(f"ℹ️ Video path updated to: {video_path}")
            else:
                logging.error(f"Video file not found: {video_path} (also tried {video_path_with_ext})")
                print(f"❌ Video file not found: {video_path} (also tried {video_path_with_ext})")
                raise ValueError(f"Video file not found: {video_path} (also tried {video_path_with_ext})")
        else:
            logging.error(f"Video file not found: {video_path}")
            print(f"❌ Video file not found: {video_path}")
            raise ValueError(f"Video file not found: {video_path}")

    if frame_number < 0:
        logging.error(f"Frame number must be non-negative: {frame_number}")
        print(f"❌ Frame number must be non-negative: {frame_number}")
        raise ValueError(f"Frame number must be non-negative: {frame_number}")

    # Extract frame
    original_img, frame_path = extract_frame(video_path, frame_number, output_dir)
    
    # Forensic checks on full frame
    logging.info("Performing compression artifact analysis")
    dct_heatmap_path = os.path.join(output_dir, f"compression_heatmap_frame_{frame_number}.png")
    compression_score = analyze_compression_artifacts(original_img, output_path=dct_heatmap_path)

    logging.info("Performing noise residual analysis")
    noise_score = analyze_noise_residuals(original_img)

    logging.info("Performing metadata anomaly detection")
    meta_flags = detect_metadata_anomalies(video_path)

    # Load and preprocess for model prediction
    _, face_img, img_array, face_coords = load_and_preprocess_image(frame_path)

    # Model prediction
    logging.info("Predicting deepfake probability")
    model_prob_raw = model.predict(img_array)[0][0]
    print(f"Deepfake probability (model): {model_prob_raw:.4f}")
    logging.info(f"Deepfake probability (model): {model_prob_raw:.4f}")

    # Adjust probabilities
    model_prob = adjust_model_bias(model_prob_raw)
    adjusted_prob = adjust_probability_with_metrics(model_prob, compression_score, noise_score)
    combined_prob = final_classification(adjusted_prob, meta_flags)
    print(f"Final deepfake probability: {combined_prob:.4f}")
    logging.info(f"Final deepfake probability: {combined_prob:.4f}")

    # Classification
    is_deepfake = combined_prob > 0.5
    if is_deepfake:
        print("⚠️ Frame is likely a deepfake (final probability > 0.5)")
        logging.warning("Frame is likely a deepfake")
    else:
        print("✅ Frame is likely real (final probability <= 0.5)")
        logging.info("Frame is likely real")

    result = {
        "frame_id": frame_number,
        "model_prob": float(model_prob),
        "forensic": {
            "compression_score": float(compression_score),
            "noise_score": float(noise_score),
            "metadata_flags": meta_flags,
            "watermark_tampered": False,
            "dct_heatmap_url": dct_heatmap_path
        },
        "combined_prob": float(combined_prob),
        "frame_url": frame_path,
        "is_deepfake": is_deepfake
    }

    # Generate Grad-CAM heatmap only if the video is a deepfake
    if is_deepfake:
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
        if not conv_layers:
            logging.error("No convolutional layer found in the model")
            print("❌ No convolutional layer found in the model")
            raise ValueError("No convolutional layer found in the model")
        layer_name = conv_layers[1] if len(conv_layers) > 1 else conv_layers[0]
        logging.info(f"Using layer: {layer_name} for Grad-CAM")
        print(f"ℹ️ Using layer: {layer_name} for Grad-CAM")
        
        heatmap = grad_cam(model, img_array, layer_name)
        superimposed_img = overlay_heatmap(heatmap, face_img, original_img, face_coords)
        heatmap_path = os.path.join(output_dir, f"gradcam_heatmap_frame_{frame_number}.png")
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, superimposed_img)
        logging.info(f"Grad-CAM heatmap saved to {heatmap_path}")
        print(f"✅ Saved Grad-CAM heatmap to {heatmap_path}")
        result["gradcam_url"] = heatmap_path
    else:
        result["gradcam_url"] = None  # No heatmap for real videos

    return result
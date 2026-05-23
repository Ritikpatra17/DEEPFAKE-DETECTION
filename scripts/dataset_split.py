import os
import sys
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm

# Add project root to path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import BASE_DIR

# Paths (relative to project root)
REAL_SOURCE = os.path.join(BASE_DIR, "processed_data", "faces")
FAKE_SOURCE = os.path.join(BASE_DIR, "processed_data", "fake_faces")
DATASET_SPLIT = os.path.join(BASE_DIR, "dataset_split")

# Load the face detection model (OpenCV DNN)
prototxt_path = os.path.join(BASE_DIR, "models", "face_detector", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Create dataset split folders
for split in ['train', 'val', 'test']:
    for category in ['real', 'fake']:
        os.makedirs(os.path.join(DATASET_SPLIT, split, category), exist_ok=True)

# Split Ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

def get_all_images(source_folder):
    """ Collect all image paths recursively from subdirectories. """
    image_paths = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def is_blurry(image, threshold=50):
    """ Check if an image is blurry based on the variance of the Laplacian. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian < threshold  # Lower threshold means more blurry images are allowed

def detect_face(image_path):
    """ Detects face in an image and filters based on size & blurriness. """
    image = cv2.imread(image_path)
    if image is None:
        return False  # Skip corrupted images

    h, w = image.shape[:2]
    
    # Skip too blurry images (reduced strictness)
    if is_blurry(image, threshold=30):  # Lowered threshold from 50 to 30
        return False

    # Perform face detection
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:  # Keep threshold high
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x, y, x2, y2 = box.astype("int")
            face_width, face_height = x2 - x, y2 - y
            
            # Accept smaller faces but still reasonable
            if face_width > 20 and face_height > 20:  # Lowered from 30×30 to 20×20
                return True  # Valid face found

    return False  # No valid face detected

def split_and_copy_images(source_folder, category):
    """ Splits and copies valid images into train/val/test folders. """
    all_images = get_all_images(source_folder)
    valid_images = [img for img in tqdm(all_images, desc=f"Filtering {category} images") if detect_face(img)]

    # Shuffle before splitting
    random.shuffle(valid_images)

    total_images = len(valid_images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    train_images = valid_images[:train_count]
    val_images = valid_images[train_count:train_count + val_count]
    test_images = valid_images[train_count + val_count:]

    print(f"✅ Found {total_images} valid {category} images: {train_count} train, {val_count} val, {len(test_images)} test.")

    for img in tqdm(train_images, desc=f"Copying {category} images to train"):
        shutil.copy(img, os.path.join(DATASET_SPLIT, 'train', category, os.path.basename(img)))
    for img in tqdm(val_images, desc=f"Copying {category} images to val"):
        shutil.copy(img, os.path.join(DATASET_SPLIT, 'val', category, os.path.basename(img)))
    for img in tqdm(test_images, desc=f"Copying {category} images to test"):
        shutil.copy(img, os.path.join(DATASET_SPLIT, 'test', category, os.path.basename(img)))

# Process real and fake images
split_and_copy_images(REAL_SOURCE, "real")
split_and_copy_images(FAKE_SOURCE, "fake")

print("✅ Dataset split completed with improved filtering.")

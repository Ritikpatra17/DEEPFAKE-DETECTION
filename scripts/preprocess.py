import os
import cv2
import shutil
from tqdm import tqdm

# Define input and output directories
INPUT_FOLDER = "C:\\Users\\NAMAN\\Deepfake-Detection\\dataset\\celeb-df-v2"  # Ensure this path is correct
OUTPUT_FOLDER = "C:\\Users\\NAMAN\\Deepfake-Detection\\processed_data\\frames"

# Create output directory if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_frames(video_path, save_dir, frame_rate=1):
    """ Extract frames from a video at a given frame rate. """
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps is None or fps == 0:
        print(f"Warning: Unable to read FPS for {video_path}, using default frame interval.")
        fps = 30  # Assuming 30 FPS if unreadable
    
    frame_interval = max(1, int(fps // frame_rate))

    frame_count = 0
    extracted_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(save_dir, f"frame_{extracted_count:04d}.jpg")
            try:
                cv2.imwrite(frame_filename, frame)
                extracted_count += 1
            except Exception as e:
                print(f"Error saving frame {extracted_count} for {video_path}: {e}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path}")

# Process all videos in the dataset folder
CATEGORIES = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]

for category in CATEGORIES:
    category_path = os.path.join(INPUT_FOLDER, category)
    if not os.path.exists(category_path):
        print(f"Warning: {category_path} does not exist. Skipping.")
        continue
    
    print(f"Processing {category}...")
    for video_name in tqdm(os.listdir(category_path)):
        if video_name.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(category_path, video_name)
            save_dir = os.path.join(OUTPUT_FOLDER, category, video_name.split('.')[0])
            try:
                extract_frames(video_path, save_dir)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

print("Preprocessing complete!")

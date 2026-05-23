import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from scripts.analyze_video import analyze_video
from config import UPLOAD_FOLDER, OUTPUT_FOLDER, MAX_UPLOAD_SIZE_MB, FLASK_HOST, FLASK_PORT
import numpy as np

app = Flask(__name__)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Custom function to convert NumPy types to Python types
def convert_to_json_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if a video file was uploaded
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files['video']
        frame_number = int(request.form.get('frame_number', 12))

        # Validate video file
        if not video_file.filename:
            return jsonify({"error": "Empty video filename"}), 400

        # Check file size
        video_file.seek(0, os.SEEK_END)
        file_size = video_file.tell()
        max_size = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_size:
            return jsonify({"error": f"Video file too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB."}), 400
        video_file.seek(0)

        # Save the uploaded video
        video_filename = f"{uuid.uuid4()}_{video_file.filename}"
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)

        # Analyze the video
        result = analyze_video(video_path, frame_number, OUTPUT_FOLDER)

        # Construct URLs for the images
        server_url = request.host_url.rstrip('/')
        if result['frame_url']:
            frame_url = result['frame_url'].replace('\\', '/')
            result['frame_url'] = f"{server_url}/{frame_url}"
        if result.get('gradcam_url'):
            gradcam_url = result['gradcam_url'].replace('\\', '/')
            result['gradcam_url'] = f"{server_url}/{gradcam_url}"
        if result['forensic'].get('dct_heatmap_url'):
            dct_heatmap_url = result['forensic']['dct_heatmap_url'].replace('\\', '/')
            result['forensic']['dct_heatmap_url'] = f"{server_url}/{dct_heatmap_url}"

        # Convert NumPy types to Python types
        result = convert_to_json_serializable(result)

        # Clean up the uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<path:path>')
def static_files(path):
    if path.startswith('analyze'):
        return jsonify({"error": "Endpoint not found"}), 404
    return send_from_directory('', path)

if __name__ == '__main__':
    app.run(debug=False, host=FLASK_HOST, port=FLASK_PORT)
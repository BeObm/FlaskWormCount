from flask import Flask, request, render_template, send_from_directory, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from utils import draw_detections
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tempfile
import io
import zipfile
import uuid
from datetime import datetime

# --- Flask Setup ---
app = Flask(__name__, static_folder='static')
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['USER_LOGS_FOLDER'] = 'static/user_logs'

# --- Ensure necessary folders exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['USER_LOGS_FOLDER'], exist_ok=True)

# --- Load YOLO model ---
model = YOLO("runs/detect/train/weights/best.pt")


def get_user_session_id():
    """Get or create a unique session ID for the user"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


def get_user_log_path(user_id):
    """Get the path for user's individual log file"""
    return os.path.join(app.config['USER_LOGS_FOLDER'], f'worm_log_{user_id}.xlsx')


def get_user_output_folder(user_id):
    """Get the output folder for user's processed images"""
    user_output_folder = os.path.join(app.config['OUTPUT_FOLDER'], user_id)
    os.makedirs(user_output_folder, exist_ok=True)
    return user_output_folder


def initialize_user_log(user_id):
    """Create log file for user if it doesn't exist"""
    log_path = get_user_log_path(user_id)
    if not os.path.isfile(log_path):
        df = pd.DataFrame(columns=["Image", "Worm Count", "Upload Time"])
        df.to_excel(log_path, index=False)
    return log_path


# --- Main Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    user_id = get_user_session_id()
    initialize_user_log(user_id)

    detections = []
    if request.method == "POST":
        uploaded_files = request.files.getlist("files[]")
        user_output_folder = get_user_output_folder(user_id)

        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_{filename}")
            uploaded_file.save(file_path)

            # Convert to BGR for OpenCV
            image = Image.open(file_path).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Save to temp file for YOLO
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, image_bgr)
                temp_path = tmp.name

            # Run YOLO detection
            results = model(temp_path)
            result = results[0]
            worm_count = len(result.boxes)

            # Draw detections and save to user's folder
            img_with_detections = draw_detections(result)
            output_filename = f"detected_{os.path.splitext(filename)[0]}_{worm_count}.jpg"
            output_path = os.path.join(user_output_folder, output_filename)
            cv2.imwrite(output_path, img_with_detections)

            # Log results to user's individual log file
            log_path = get_user_log_path(user_id)
            df = pd.read_excel(log_path)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.loc[len(df)] = [output_filename, worm_count, current_time]
            df.to_excel(log_path, index=False)

            # Append for front-end display
            detections.append((output_filename, worm_count))

            # Clean up temp files
            os.unlink(temp_path)
            os.unlink(file_path)

    return render_template("index.html", detections=detections)


# Download user's individual worm_log.xlsx
@app.route("/download/log")
def download_log():
    user_id = get_user_session_id()
    log_path = get_user_log_path(user_id)

    if not os.path.exists(log_path):
        initialize_user_log(user_id)

    return send_file(
        log_path,
        as_attachment=True,
        download_name=f"worm_log_{user_id[:8]}.xlsx"
    )


# Download user's detected output images as ZIP
@app.route("/download/images")
def download_images():
    user_id = get_user_session_id()
    user_output_folder = get_user_output_folder(user_id)

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in os.listdir(user_output_folder):
            if filename.endswith((".jpg", ".png")):
                file_path = os.path.join(user_output_folder, filename)
                zf.write(file_path, arcname=filename)

    memory_file.seek(0)
    return send_file(
        memory_file,
        as_attachment=True,
        download_name=f"detected_worms_{user_id[:8]}.zip",
        mimetype='application/zip'
    )


# --- Serve output images (with user isolation) ---
@app.route("/outputs/<filename>")
def output_file(filename):
    user_id = get_user_session_id()
    user_output_folder = get_user_output_folder(user_id)
    return send_from_directory(user_output_folder, filename)


# --- Clear user session (optional endpoint for testing) ---
@app.route("/clear_session")
def clear_session():
    session.clear()
    return redirect(url_for('index'))


# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)
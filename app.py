from flask import Flask, request, render_template, send_from_directory, send_file, redirect, url_for
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


# --- Flask Setup ---
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['LOG_FILE'] = os.path.join(app.config['OUTPUT_FOLDER'], 'worm_log.xlsx')

# --- Ensure necessary folders exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Create log file if not present ---
if not os.path.isfile(app.config['LOG_FILE']):
    df = pd.DataFrame(columns=["Image", "Worm Count"])
    df.to_excel(app.config['LOG_FILE'], index=False)

# --- Load YOLO model ---
model = YOLO("runs/detect/train/weights/best.pt")


# --- Main Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    detections = []
    if request.method == "POST":
        uploaded_files = request.files.getlist("files[]")
        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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

            # Draw detections and save
            img_with_detections = draw_detections(result)
            output_filename = f"detected_{os.path.splitext(filename)[0]}_{worm_count}.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, img_with_detections)

            # Log results
            df = pd.read_excel(app.config['LOG_FILE'])
            df.loc[len(df)] = [output_filename, worm_count]
            df.to_excel(app.config['LOG_FILE'], index=False)

            # Append for front-end display
            detections.append((output_filename, worm_count))

    return render_template("index.html", detections=detections)

# Download worm_log.xlsx
@app.route("/download/log")
def download_log():
    return send_file(
        app.config['LOG_FILE'],
        as_attachment=True,
        download_name="worm_log.xlsx"
    )

# Download all detected output images as ZIP
@app.route("/download/images")
def download_images():
    output_dir = app.config['OUTPUT_FOLDER']
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in os.listdir(output_dir):
            if filename.endswith((".jpg", ".png")):
                file_path = os.path.join(output_dir, filename)
                zf.write(file_path, arcname=filename)

    memory_file.seek(0)
    return send_file(
        memory_file,
        as_attachment=True,
        download_name="detected_worms.zip",
        mimetype='application/zip'
    )

# --- Serve output images ---
@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)

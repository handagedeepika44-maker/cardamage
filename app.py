# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "static/outputs"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# change to your custom weights if you have them
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET", "POST"])
def index():
    filename = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            in_path = os.path.join(app.config["UPLOAD_FOLDER"], "upload_" + fname)
            file.save(in_path)

            # Read and run inference
            img_bgr = cv2.imread(in_path)
            if img_bgr is None:
                return "Failed to read uploaded image", 500
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = model.predict(img_rgb, imgsz=640, conf=0.25, verbose=False)
            annotated = results[0].plot()
            out_name = "detected_" + fname
            out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
            cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            filename = out_name
            return render_template("index.html", filename=filename)
    return render_template("index.html", filename=None)

if __name__ == "__main__":
    # Use 0.0.0.0 if you want to access from other devices on the network
    app.run(host="127.0.0.1", port=5000, debug=True)
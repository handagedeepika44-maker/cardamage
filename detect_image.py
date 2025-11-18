# detect_image.py
import os
import sys
from ultralytics import YOLO
import cv2

# Path to image to test
IMAGE_PATH = "test_images/car1.jpg"  # change to your image path
OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose model: replace "yolov8n.pt" with "Weights/best.pt" if you trained a model
MODEL_PATH = "yolov8n.pt"

def main(image_path=IMAGE_PATH):
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print("Loading model...")
    model = YOLO(MODEL_PATH)  # will download yolov8n.pt if missing

    print(f"Running inference on {image_path} ...")
    results = model(image_path, imgsz=640, conf=0.25)  # returns a list of Results

    # results[0].plot() returns an annotated image (numpy array)
    annotated = results[0].plot()
    filename = os.path.basename(image_path)
    out_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
    cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to: {out_path}")

if __name__ == "__main__":
    main()

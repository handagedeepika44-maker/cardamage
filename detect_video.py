# detect_video.py
import os
import cv2
from ultralytics import YOLO

INPUT_VIDEO = "Media/cardent.mp4"  # change to your video file
OUTPUT_DIR = "static/outputs"
OUT_VIDEO = os.path.join(OUTPUT_DIR, "detected_video.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "yolov8n.pt"  # or "Weights/best.pt"

def main():
    if not os.path.isfile(INPUT_VIDEO):
        print(f"Video not found: {INPUT_VIDEO}")
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    frame_count = 0
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # ultralytics expects RGB input if you pass ndarray; convert BGR->RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()  # RGB annotated image
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Saved annotated video to: {OUT_VIDEO}")

if __name__ == "__main__":
    main()
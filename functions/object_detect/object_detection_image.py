import cv2
import torch
import numpy as np
import requests
from ultralytics import YOLO
import time
from pathlib import Path

def object_detection_image(path: str = None, url: str = None, model_id: str = "yolov8l.pt", output_dir: str = "."):

    # GPU Detection
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ GPU:", torch.cuda.get_device_name(0))
        imgsz=1280
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon GPU")
        imgsz=1280
    else:
        device = "cpu"
        print("Using CPU as GPU is not available")
        imgsz=640
        model_id = "yolov8s.pt"


    # Load model ON GPU
    model = YOLO(model_id)
    model.to(device)
    if device == "cuda":
        model.fuse()   # speedup


    # Load image from URL or path
    if url is not None:
        resp = requests.get(url, timeout=10)
        image_np = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    elif path is not None:
        image = cv2.imread(path)
    
    else:
        url = "https://ultralytics.com/images/bus.jpg"
        resp = requests.get(url, timeout=10)
        image_np = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    assert image is not None, "❌ Image decode failed"


    # Run model
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=0.25,
        iou=0.5,
        device=device,
        verbose=True
    )


    # Draw & save
    annotated = results[0].plot()
    timestamp = int(time.time())

    if not output_dir.endswith("/"):
        output_dir += "/"
    
    # Creating output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = f"{output_dir}/yolo_image_{timestamp}.png"

    cv2.imwrite(output_path, annotated)

    print("✅ Detection complete. Image saved at:", output_path)

    return annotated

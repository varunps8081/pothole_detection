from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import os
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Define the path to the YOLO model (Ensure this is the correct path)
MODEL_PATH = r"C:\Users\user\Downloads\best.pt"  # Use raw string (r"") to avoid Unicode errors

# Load YOLO model
model = YOLO(MODEL_PATH)

# Ensure the "uploads" folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to Pothole Detection API"}

@app.get("/favicon.ico")
async def favicon():
    return {}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video file and performs pothole detection."""
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded video
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video for pothole detection
    output_path = process_video(file_path)

    return {"message": "Processing complete", "output_video": output_path}

def process_video(video_path):
    """Detects potholes in a video using YOLO and saves the processed video."""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Error opening video file"}

    output_video_path = video_path.replace(".mp4", "_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = float(box.conf[0])  # Confidence score
                
                # Draw bounding box and label
                label = f"Pothole: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return output_video_path

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

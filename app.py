import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# ---------------------- CONFIGURE UI ----------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Sidebar UI
st.sidebar.title("🔍 YOLO Object Detection")
st.sidebar.markdown("Control the real-time object detection system.")

# Load YOLO Model (replace with your model path)
MODEL_PATH = "best.pt"  # Change this to your local model file
model = YOLO(MODEL_PATH)

# Session state to manage app flow
if "video_running" not in st.session_state:
    st.session_state.video_running = False

# Start & Stop buttons in Sidebar
start_button = st.sidebar.button("▶️ Start Video", use_container_width=True)
stop_button = st.sidebar.button("⏹ Stop Video", use_container_width=True)

# ---------------------- VIDEO PROCESSING ----------------------
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

def run_video():
    frame_placeholder = st.empty()  # Placeholder for video frames
    fps_placeholder = st.sidebar.empty()  # Placeholder for FPS display

    prev_time = 0

    while st.session_state.video_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video. Check your camera connection.")
            break

        # Mirror the frame for a natural webcam feel
        frame = cv2.flip(frame, 1)

        # Run YOLO inference (disable logging)
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Draw bounding boxes
        for detection in detections:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            conf = detection.conf.item()
            class_id = int(detection.cls.item())
            label = f"{model.names[class_id]}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        fps_placeholder.markdown(f"**⏳ FPS:** {fps:.2f}")

        # Convert frame to RGB and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_container_width=True)

        # Stop if button is pressed
        if not st.session_state.video_running:
            break

# ---------------------- HANDLE BUTTON CLICKS ----------------------
if start_button:
    st.session_state.video_running = True
    run_video()

if stop_button:
    st.session_state.video_running = False
    cap.release()
    st.sidebar.success("✅ Video Stopped")

# Release resources when the script stops
if not st.session_state.video_running:
    cap.release()

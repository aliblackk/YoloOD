import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ---------------------- CONFIGURE UI ----------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.sidebar.title("🔍 YOLO Object Detection with Storage")

# Create a folder for storing captured images/videos
SAVE_DIR = "storage"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load YOLO Model (replace with your model path)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Session state
if "video_running" not in st.session_state:
    st.session_state.video_running = False

start_button = st.sidebar.button("▶️ Start Video", use_container_width=True)
stop_button = st.sidebar.button("⏹ Stop Video", use_container_width=True)
capture_button = st.sidebar.button("📸 Capture Image", use_container_width=True)

cap = cv2.VideoCapture(0)

# ---------------------- VIDEO PROCESSING ----------------------
def run_video():
    frame_placeholder = st.empty()
    fps_placeholder = st.sidebar.empty()

    prev_time = 0

    while st.session_state.video_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video.")
            break

        frame = cv2.flip(frame, 1)
        results = model(frame, verbose=False)
        detections = results[0].boxes

        for detection in detections:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            conf = detection.conf.item()
            class_id = int(detection.cls.item())
            label = f"{model.names[class_id]}: {conf:.2f}"
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        fps_placeholder.markdown(f"**⏳ FPS:** {fps:.2f}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_container_width=True)

        if not st.session_state.video_running:
            break

# ---------------------- CAPTURE IMAGE ----------------------
def capture_image():
    ret, frame = cap.read()
    if ret:
        filename = f"{SAVE_DIR}/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        st.sidebar.success(f"✅ Image saved: {filename}")

# ---------------------- HANDLE BUTTON CLICKS ----------------------
if start_button:
    st.session_state.video_running = True
    run_video()

if stop_button:
    st.session_state.video_running = False
    cap.release()
    st.sidebar.success("✅ Video Stopped")

if capture_button:
    capture_image()

if not st.session_state.video_running:
    cap.release()

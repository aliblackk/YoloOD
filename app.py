import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------------- CONFIGURE UI ----------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.sidebar.title("🔍 YOLO Object Detection with Storage")

# Create a folder for storing captured images
SAVE_DIR = "storage"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load YOLO Model (replace with your model path)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# ---------------------- VIDEO PROCESSING ----------------------
class YOLOVideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array
        results = model(img, verbose=False)  # Run YOLO detection
        
        for detection in results[0].boxes:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            conf = detection.conf.item()
            class_id = int(detection.cls.item())
            label = f"{model.names[class_id]}: {conf:.2f}"

            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------- START VIDEO STREAM ----------------------
st.sidebar.markdown("### 🎥 Live Stream")
webrtc_ctx = webrtc_streamer(
    key="yolo-stream",
    video_transformer_factory=YOLOVideoProcessor,
    async_processing=True
)

# ---------------------- CAPTURE IMAGE ----------------------
def capture_image():
    if webrtc_ctx.video_receiver:
        frame = webrtc_ctx.video_receiver.last_frame
        if frame is not None:
            img = frame.to_ndarray(format="bgr24")
            filename = f"{SAVE_DIR}/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, img)
            st.sidebar.success(f"✅ Image saved: {filename}")
        else:
            st.sidebar.error("❌ No frame captured!")

if st.sidebar.button("📸 Capture Image"):
    capture_image()

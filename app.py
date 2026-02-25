import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.title("YOLOv8 Object Detection")

# Load model without cache
model = YOLO("runs/detect/train/weights/best.pt")

uploaded_video = st.file_uploader("Upload a video", type=["mp4","avi","mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")
    cap.release()

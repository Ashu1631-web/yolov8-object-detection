import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Vehicle Detection System", layout="centered")

st.title("🚗 Vehicle Detection using YOLOv8")
st.write("Select an image from dataset or upload your own image.")

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train3/weights/best.pt")

model = load_model()

# Confidence slider
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# -----------------------------
# OPTION 1: Select from Dataset
# -----------------------------
st.subheader("📂 Select Image from Dataset")

image_folder = "data/test/images"

if os.path.exists(image_folder):

    image_files = os.listdir(image_folder)

    if len(image_files) > 0:

        selected_image = st.selectbox("Choose an image", image_files)

        if selected_image:
            image_path = os.path.join(image_folder, selected_image)
            image = Image.open(image_path)

            st.image(image, caption="Original Image", use_container_width=True)

            # Run detection
            results = model(image, conf=confidence)

            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            st.image(annotated_frame, caption="Detected Image", use_container_width=True)

            # Show detection details
            st.write("Detected Objects:")
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    class_name = model.names[cls_id]
                    st.write(f"{class_name} ({conf_score:.2f})")
            else:
                st.write("No objects detected.")

    else:
        st.warning("No images found in dataset folder.")

else:
    st.error("Dataset folder not found!")

# -----------------------------
# OPTION 2: Upload Image
# -----------------------------
st.subheader("📤 Upload Your Own Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(image, conf=confidence)

    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.image(annotated_frame, caption="Detected Image", use_container_width=True)

    st.write("Detected Objects:")
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = model.names[cls_id]
            st.write(f"{class_name} ({conf_score:.2f})")
    else:
        st.write("No objects detected.")

st.success("App Ready ✅")
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# Run detection on test images folder
results = model("data/test/images", save=True)

print("Detection Completed Successfully")
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np

# ------------------------------
# YOLOv8 Object Detection Model
# ------------------------------
yolo_model = YOLO("yolov8n.pt")  # lightweight version

classification_model = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    top_k=3
)


# ------------------------------
# Emotion Model
# ------------------------------
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=5
)

def analyze_image(image):
    classification_results = classification_model(image)

for r in classification_results:
    if r["score"] > 0.2:
        object_results.append({
            "label": r["label"],
            "score": r["score"]
        })


    # Convert PIL to numpy
    image_np = np.array(image)

    # ---------------- OBJECT DETECTION ----------------
    results = yolo_model(image_np)

    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            confidence = float(box.conf[0])

            if confidence > 0.4:
                detected_objects.append({
                    "label": label,
                    "score": confidence
                })

    # Remove duplicates
    unique_objects = {}
    for obj in detected_objects:
        if obj["label"] not in unique_objects:
            unique_objects[obj["label"]] = obj["score"]

    object_results = [
        {"label": k, "score": v}
        for k, v in unique_objects.items()
    ]

    # ---------------- EMOTION DETECTION ----------------
    try:
        emotion_results = emotion_classifier(image)
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    except:
        dominant_emotion = "Not Detected"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores


from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np


# YOLOv8 Lightweight Model

yolo_model = YOLO("yolov8n.pt")


# Image Classification Model (Fallback / Extra Info)

classification_model = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    top_k=3
)

# Emotion Model

emotion_model = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=5
)

def analyze_image(image):

    image_np = np.array(image)

    detected_objects = []

    #YOLO OBJECT DETECTION
    try:
        results = yolo_model(image_np)

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
    except:
        pass

    # CLASSIFICATION FALLBACK 
    try:
        classification_results = classification_model(image)

        for r in classification_results:
            if r["score"] > 0.25:
                detected_objects.append({
                    "label": r["label"],
                    "score": r["score"]
                })
    except:
        pass

    # Remove duplicates
    unique = {}
    for obj in detected_objects:
        if obj["label"] not in unique:
            unique[obj["label"]] = obj["score"]

    object_results = [
        {"label": k, "score": v}
        for k, v in unique.items()
    ]

    # EMOTION DETECTION 
    try:
        emotion_results = emotion_model(image)
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    except:
        dominant_emotion = "Not Detected"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores

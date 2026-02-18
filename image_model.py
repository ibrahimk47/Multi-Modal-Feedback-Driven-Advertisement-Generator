from ultralytics import YOLO
from transformers import pipeline
import numpy as np

yolo_model = None
emotion_model = None

def load_models():
    global yolo_model, emotion_model

    if yolo_model is None:
        yolo_model = YOLO("yolov8n.pt")

    if emotion_model is None:
        emotion_model = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            top_k=3
        )

def analyze_image(image):

    load_models()

    image_np = np.array(image)

    # OBJECT DETECTION
    results = yolo_model(image_np)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if confidence > 0.4:
                detected_objects.append({
                    "label": label,
                    "score": confidence
                })

    # EMOTION DETECTION
    emotion_results = emotion_model(image)
    dominant_emotion = emotion_results[0]["label"]
    emotion_scores = {r["label"]: r["score"] for r in emotion_results}

    return detected_objects, dominant_emotion, emotion_scores

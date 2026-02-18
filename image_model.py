import os
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from huggingface_hub import login

# ------------------------------------------------
# OPTIONAL: HuggingFace Auth (for faster download)
# ------------------------------------------------
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ------------------------------------------------
# Global Model Variables (Lazy Loading)
# ------------------------------------------------
yolo_model = None
emotion_model = None

# ------------------------------------------------
# Load Models Only When Needed
# ------------------------------------------------
def load_models():
    global yolo_model, emotion_model

    if yolo_model is None:
        yolo_model = YOLO("yolov8n.pt")  # lightweight nano model

    if emotion_model is None:
        emotion_model = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            top_k=3
        )

# ------------------------------------------------
# Main Image Analysis Function
# ------------------------------------------------
def analyze_image(image):

    load_models()

    image_np = np.array(image)

    detected_objects = []

    # ---------------- YOLO OBJECT DETECTION ----------------
    try:
        results = yolo_model(image_np)

        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                if confidence > 0.4:
                    detected_objects.append({
                        "label": label,
                        "score": confidence
                    })
    except Exception as e:
        print("YOLO error:", e)

    # Remove duplicate labels
    unique = {}
    for obj in detected_objects:
        if obj["label"] not in unique:
            unique[obj["label"]] = obj["score"]

    object_results = [
        {"label": k, "score": v}
        for k, v in unique.items()
    ]

    # ---------------- EMOTION DETECTION ----------------
    try:
        emotion_results = emotion_model(image)
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    except Exception as e:
        print("Emotion error:", e)
        dominant_emotion = "Not Detected"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores



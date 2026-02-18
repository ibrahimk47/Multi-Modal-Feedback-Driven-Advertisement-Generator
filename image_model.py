from transformers import pipeline
import numpy as np

# --------------------------------------------
# Lightweight Image Classification Model
# --------------------------------------------
object_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    top_k=5
)

# --------------------------------------------
# Lightweight Emotion Model
# --------------------------------------------
emotion_model = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=3
)

# --------------------------------------------
# Image Analysis Function
# --------------------------------------------
def analyze_image(image):

    detected_objects = []

    # -------- OBJECT CLASSIFICATION --------
    try:
        results = object_classifier(image)

        for r in results:
            if r["score"] > 0.15:  # Lower threshold for multiple outputs
                detected_objects.append({
                    "label": r["label"],
                    "score": r["score"]
                })

    except Exception as e:
        print("Object classification error:", e)

    # Remove duplicates
    unique = {}
    for obj in detected_objects:
        if obj["label"] not in unique:
            unique[obj["label"]] = obj["score"]

    object_results = [
        {"label": k, "score": v}
        for k, v in unique.items()
    ]

    # -------- EMOTION DETECTION --------
    try:
        emotion_results = emotion_model(image)
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    except Exception as e:
        print("Emotion error:", e)
        dominant_emotion = "Neutral"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores

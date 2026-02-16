from transformers import pipeline
from PIL import Image

# Object classification model
object_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    top_k=5
)

# Facial emotion classification model
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=5
)

def analyze_image(image):

    #  OBJECT CLASSIFICATION 
    object_results = object_classifier(image)

    # EMOTION DETECTION 
    emotion_results = emotion_classifier(image)

    dominant_emotion = emotion_results[0]["label"]
    emotion_scores = {r["label"]: r["score"] for r in emotion_results}

    return object_results, dominant_emotion, emotion_scores


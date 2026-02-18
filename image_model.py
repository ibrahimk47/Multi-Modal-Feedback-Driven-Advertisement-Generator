from transformers import pipeline
from PIL import Image

# ------------------------------
# REAL OBJECT DETECTION (DETR)
# ------------------------------
object_detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50"
)

# ------------------------------
# FACE EMOTION MODEL
# ------------------------------
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=5
)

def analyze_image(image):

    # ---------------- OBJECT DETECTION ----------------
    object_results = object_detector(image)

    # ---------------- EMOTION DETECTION ----------------
    try:
        emotion_results = emotion_classifier(image)
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    except:
        dominant_emotion = "Emotion Not Detected"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores

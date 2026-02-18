from transformers import pipeline

# Lightweight Image Classification Model
object_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    top_k=5
)

# Facial Emotion Model
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=5
)

def analyze_image(image):

    # ---------------- OBJECT PREDICTION ----------------
    object_results = object_classifier(image)

    # Filter low-confidence predictions
    object_results = [
        r for r in object_results if r["score"] > 0.2
    ]

    if not object_results:
        object_results = object_classifier(image)[:3]

    # ---------------- EMOTION PREDICTION ----------------
    emotion_results = emotion_classifier(image)

    emotion_results = [
        r for r in emotion_results if r["score"] > 0.2
    ]

    if emotion_results:
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    else:
        dominant_emotion = "Neutral"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores

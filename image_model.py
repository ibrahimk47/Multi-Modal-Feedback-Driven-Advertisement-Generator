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

    # Get top 5 predictions
    object_results = object_classifier(image)

    # Keep top 5 confidently predicted labels
    object_results = sorted(object_results, key=lambda x: x["score"], reverse=True)[:5]

    # Clean labels
    cleaned_objects = []
    for r in object_results:
        label = r["label"]
        score = r["score"]
        if score > 0.1:  # lower threshold to allow multiple objects
            cleaned_objects.append({
                "label": label,
                "score": score
            })

    # Emotion Detection
    emotion_results = emotion_classifier(image)

    emotion_results = sorted(emotion_results, key=lambda x: x["score"], reverse=True)

    dominant_emotion = emotion_results[0]["label"]
    emotion_scores = {r["label"]: r["score"] for r in emotion_results}

    return cleaned_objects, dominant_emotion, emotion_scores

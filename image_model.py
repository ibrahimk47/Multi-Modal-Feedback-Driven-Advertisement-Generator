from transformers import pipeline

object_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    top_k=5
)

emotion_model = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=3
)

def analyze_image(image):

    detected_objects = []

    try:
        results = object_classifier(image)

        for r in results:
            if r["score"] > 0.15:
                detected_objects.append({
                    "label": r["label"],
                    "score": r["score"]
                })
    except:
        pass

    unique = {}
    for obj in detected_objects:
        if obj["label"] not in unique:
            unique[obj["label"]] = obj["score"]

    object_results = [
        {"label": k, "score": v}
        for k, v in unique.items()
    ]

    try:
        emotion_results = emotion_model(image)
        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}
    except:
        dominant_emotion = "Neutral"
        emotion_scores = {}

    return object_results, dominant_emotion, emotion_scores

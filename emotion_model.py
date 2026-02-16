from transformers import pipeline

# Emotion detection model
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def detect_emotion(text):
    result = emotion_pipeline(text)[0][0]
    return result["label"], round(result["score"], 3)

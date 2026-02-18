from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

# REAL OBJECT DETECTION (DETR)

object_detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50"
)


# FACE EMOTION MODEL

emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    top_k=5
)


# FACE DETECTOR (OpenCV)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def analyze_image(image):

    # Convert PIL to numpy
    image_np = np.array(image)

    # OBJECT DETECTION 
    object_results = object_detector(image)

    #  FACE DETECTION 
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    dominant_emotion = "No Face Detected"
    emotion_scores = {}

    if len(faces) > 0:

        # Take first detected face
        (x, y, w, h) = faces[0]

        face_crop = image_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_crop)

        emotion_results = emotion_classifier(face_pil)

        dominant_emotion = emotion_results[0]["label"]
        emotion_scores = {r["label"]: r["score"] for r in emotion_results}

    return object_results, dominant_emotion, emotion_scores

import streamlit as st
from PIL import Image
from utils import clean_text
from sentiment_model import analyze_sentiment
from emotion_model import detect_emotion
from image_model import analyze_image
from ad_generator import generate_ad

st.set_page_config(page_title="Multi-Modal Advertisement Generator", layout="wide")

st.title("Multi-Modal Feedback-Driven Advertisement Generation")

page = st.sidebar.radio("Navigation", ["Text Analysis", "Image Analysis"])

# ===================================================
# TEXT ANALYSIS
# ===================================================
if page == "Text Analysis":

    st.header("Text Sentiment & Emotion Analysis")

    user_input = st.text_area("Enter Customer Feedback")

    if st.button("Analyze Text"):

        if user_input.strip() != "":

            processed = clean_text(user_input)

            sentiment, sent_score = analyze_sentiment(processed)
            emotion, emo_score = detect_emotion(processed)

            ad = generate_ad(
                text_sentiment=sentiment,
                text_emotion=emotion,
                user_feedback=user_input,
                image_emotion=None,
                detected_object=None
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Analysis Results")
                st.write(f"Sentiment: {sentiment}")
                st.write(f"Emotion: {emotion}")

            with col2:
                st.subheader("Advertisement Suggestion")
                st.success(ad)

        else:
            st.warning("Please enter feedback.")

# ===================================================
# IMAGE ANALYSIS
# ===================================================
elif page == "Image Analysis":

    st.header("Image-Based Advertisement Suggestion")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze Image"):

            object_results, dominant_emotion, emotion_scores = analyze_image(image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Detected Categories")

                if object_results:
                    for obj in object_results:
                        st.write(f"- {obj['label']} ({obj['score']:.2f})")
                else:
                    st.write("No objects detected.")

                st.write(f"Dominant Emotion: {dominant_emotion}")

            with col2:

                detected_object_string = ", ".join(
                    [obj["label"] for obj in object_results]
                )

                ad = generate_ad(
                    text_sentiment="NEUTRAL",
                    text_emotion=None,
                    user_feedback="Visual Interaction",
                    image_emotion=dominant_emotion,
                    detected_object=detected_object_string
                )

                st.subheader("Advertisement Suggestion")
                st.success(ad)

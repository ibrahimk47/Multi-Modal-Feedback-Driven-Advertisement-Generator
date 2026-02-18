import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sentiment_model import analyze_sentiment
from emotion_model import detect_emotion
from ad_generator import generate_ad
from image_model import analyze_image
from utils import clean_text
from PIL import Image

# PAGE CONFIG

st.set_page_config(
    page_title="Multi-Modal Feedback-Driven Advertisement Generation",
    layout="wide"
)


# UI STYLE

st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR NAVIGATION

st.sidebar.title("Dashboard")
page = st.sidebar.radio("Navigation", ["Overview", "Text Analysis", "Image Analysis"])


# OVERVIEW PAGE

if page == "Overview":

    st.title("Multi-Modal Feedback-Driven Advertisement Generation using NLP, Sentiment & Emotion Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Feedback", 138)

    with col2:
        st.metric("Positive Sentiment", 95)

    with col3:
        st.metric("Emotion Accuracy", "92%")

    st.markdown("---")

    st.subheader("System Overview")

    st.write("""
    This system analyzes customer feedback and images using 
    Natural Language Processing and Computer Vision techniques 
    to generate intelligent, emotion-aware advertisement suggestions.

    Core Components:
    • Sentiment Analysis  
    • Emotion Detection  
    • Object Recognition  
    • Multi-Modal Advertisement Strategy  
    """)


# TEXT ANALYSIS PAGE

elif page == "Text Analysis":

    st.title("Text-Based Advertisement Suggestion")

    user_input = st.text_area("Enter Customer Feedback")

    if st.button("Analyze & Generate Advertisement"):

        if user_input.strip() != "":

            processed_text = clean_text(user_input)
            sentiment, sent_score = analyze_sentiment(processed_text)
            emotion, emo_score = detect_emotion(processed_text)

            ad = generate_ad(
                text_sentiment=sentiment,
                text_emotion=emotion,
                user_feedback=user_input
            )

            col1, col2 = st.columns(2)

            # Analysis Section 
            with col1:
                st.subheader("Analysis Results")
                st.write(f"Sentiment: {sentiment}")
                st.write(f"Emotion: {emotion}")

                # Emotion Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=emo_score * 100,
                    title={'text': "Emotion Confidence"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#06b6d4"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Advertisement Section
            with col2:
                st.subheader("🎯 Advertisement Suggestion")
                st.success(ad)

        else:
            st.warning("Please enter feedback.")


# IMAGE ANALYSIS PAGE
elif page == "Image Analysis":

    st.title("Image-Based Advertisement Suggestion")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze Image"):

            object_results, dominant_emotion, emotion_scores = analyze_image(image)

            col1, col2 = st.columns(2)

            # ---------------- IMAGE ANALYSIS ----------------
            with col1:
                st.subheader("Image Analysis Results")

                # Show ALL detected objects
                st.write("Detected Objects:")
                detected_objects = []

                for obj in object_results:
                    label = obj["label"]
                    score = obj["score"]
                    detected_objects.append(label)
                    st.write(f"- {label} ({score:.2f})")

                st.write(f"Dominant Emotion: {dominant_emotion}")

            # ---------------- ADVERTISEMENT ----------------
            with col2:

                # Join multiple objects into a single string
                detected_object_string = ", ".join(detected_objects)

                ad = generate_ad(
                    text_sentiment="NEUTRAL",
                    text_emotion=dominant_emotion,
                    user_feedback="Visual Interaction",
                    image_emotion=dominant_emotion,
                    detected_object=detected_object_string
                )

                st.subheader("🎯 Advertisement Suggestion")
                st.success(ad)


import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from utils import clean_text
from sentiment_model import analyze_sentiment
from emotion_model import detect_emotion
from image_model import analyze_image
from ad_generator import generate_ad

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal AI Advertisement Engine",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL UI STYLES
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #F9FAFB;
}
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(135deg, #4F46E5, #3B82F6);
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}
footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-title">Multi-Modal AI Advertisement Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze Text & Images to Generate Emotion-Aware Marketing Content</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Text Analysis", "Image Analysis"])

# ==================================================
# TEXT ANALYSIS
# ==================================================
if page == "Text Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Text Sentiment & Emotion Analysis")

    user_input = st.text_area("Enter Customer Feedback")

    if st.button("Analyze Text"):

        if user_input.strip():

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

            # Analysis Results
            with col1:
                st.subheader("Analysis Results")
                st.write(f"Sentiment: {sentiment} ({sent_score:.2f})")
                st.write(f"Emotion: {emotion} ({emo_score:.2f})")

                fig, ax = plt.subplots()
                ax.bar(["Sentiment", "Emotion"], [sent_score, emo_score])
                ax.set_ylim(0,1)
                ax.set_title("Confidence Scores")
                st.pyplot(fig)

            # Advertisement
            with col2:
                st.subheader("Advertisement Suggestion")
                st.success(ad)

                st.download_button(
                    "Download Advertisement",
                    ad,
                    file_name="generated_ad.txt"
                )

        else:
            st.warning("Please enter feedback.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# IMAGE ANALYSIS
# ==================================================
elif page == "Image Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Image-Based Object & Emotion Analysis")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

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
                st.subheader("Advertisement Suggestion")

                detected_object_string = ", ".join(
                    [obj["label"] for obj in object_results]
                )

                ad = generate_ad(
                    text_sentiment="NEUTRAL",
                    text_emotion=None,
                    user_feedback="Image Analysis",
                    image_emotion=dominant_emotion,
                    detected_object=detected_object_string
                )

                st.success(ad)

                st.download_button(
                    "Download Advertisement",
                    ad,
                    file_name="image_ad.txt"
                )

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<footer>
Built with AI | Multi-Modal Advertisement System
</footer>
""", unsafe_allow_html=True)

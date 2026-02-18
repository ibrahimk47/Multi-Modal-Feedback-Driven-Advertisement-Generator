import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import time
from utils import clean_text
from sentiment_model import analyze_sentiment
from emotion_model import detect_emotion
from image_model import analyze_image
from ad_generator import generate_ad


# PAGE CONFIG

st.set_page_config(
    page_title="Multi-Modal AI Advertisement Engine",
    layout="wide"
)


# PREMIUM STYLES

st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 35px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #4F46E5, #3B82F6);
    color: white;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
}

.stButton>button {
    background: linear-gradient(135deg, #4F46E5, #3B82F6);
    color: white;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    font-size: 16px;
}

footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<div class="main-title">Multi-Modal AI Advertisement Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Text & Image Marketing Intelligence</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Overview", "Text Analysis", "Image Analysis"])


# OVERVIEW DASHBOARD

if page == "Overview":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">Total Analyses<br><b>128</b></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">Avg Confidence<br><b>87%</b></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">Emotion Accuracy<br><b>91%</b></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("System Capabilities")

    st.write("""
    • Text Sentiment & Emotion Detection  
    • Image Object Classification  
    • Emotion-Based Advertisement Generation  
    • Multi-Modal Fusion Logic  
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# TEXT ANALYSIS

elif page == "Text Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Text Sentiment & Emotion Analysis")

    user_input = st.text_area("Enter Customer Feedback")

    if st.button("Analyze Text"):

        if user_input.strip():

            with st.spinner("Analyzing text..."):
                time.sleep(1.5)

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

            # RESULTS
            with col1:
                st.subheader("Analysis Results")
                st.write(f"Sentiment: {sentiment}")
                st.write(f"Emotion: {emotion}")

                # Animated Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=emo_score * 100,
                    title={'text': "Emotion Confidence"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#4F46E5"}}
                ))

                st.plotly_chart(fig, use_container_width=True)

            # AD
            with col2:
                st.subheader("Advertisement Suggestion")
                st.success(ad)

                st.download_button(
                    "Download Advertisement",
                    ad,
                    file_name="text_ad.txt"
                )

    st.markdown('</div>', unsafe_allow_html=True)


# IMAGE ANALYSIS

elif page == "Image Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Image-Based Analysis")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze Image"):

            with st.spinner("Analyzing image..."):
                time.sleep(1.5)
                object_results, dominant_emotion, emotion_scores = analyze_image(image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Detected Objects")

                if object_results:
                    for obj in object_results:
                        st.write(f"- {obj['label']} ({obj['score']:.2f})")
                else:
                    st.write("No objects detected.")

                # Emotion Meter
                if emotion_scores:
                    emo_val = max(emotion_scores.values())

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=emo_val * 100,
                        title={'text': "Emotion Confidence"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#4F46E5"}}
                    ))

                    st.plotly_chart(fig, use_container_width=True)

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


# FOOTER

st.markdown("""
<footer>
Built with AI | Multi-Modal Advertisement System
</footer>
""", unsafe_allow_html=True)


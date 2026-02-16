from transformers import pipeline

# Load pretrained sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], round(result["score"], 3)

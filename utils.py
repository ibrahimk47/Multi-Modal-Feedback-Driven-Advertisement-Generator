import re
import nltk


# NLTK Setup for Streamlit Cloud (Python 3.13)

def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

# Download resources at startup
download_nltk_resources()



# Text Cleaning Function

def clean_text(text):
    """
    Cleans and preprocesses text input for NLP models.
    """

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        download_nltk_resources()
        tokens = nltk.word_tokenize(text)

    return " ".join(tokens)


import re
import nltk

# -------------------------------------------------
# Safe NLTK Setup for Local + Streamlit Cloud
# -------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# -------------------------------------------------
# Text Cleaning Function
# -------------------------------------------------
def clean_text(text):
    """
    Cleans and preprocesses text input for NLP models.
    """

    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize safely
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        tokens = nltk.word_tokenize(text)

    # Join tokens back
    cleaned_text = " ".join(tokens)

    return cleaned_text

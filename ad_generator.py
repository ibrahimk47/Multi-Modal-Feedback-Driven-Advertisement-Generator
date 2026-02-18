def generate_ad(text_sentiment, text_emotion, user_feedback,
                image_emotion, detected_object):

    strategy = "Engagement Strategy"

    if text_sentiment == "NEGATIVE":
        strategy = "Recovery Strategy"
    elif text_sentiment == "POSITIVE":
        strategy = "Upselling Strategy"

    ad = f"""
    Based on analysis, we suggest a {strategy}.
    """

    if detected_object:
        ad += f"\nFocus product category: {detected_object}."

    if image_emotion:
        ad += f"\nTarget emotion tone: {image_emotion}."

    ad += "\nCraft emotionally aligned marketing content."

    return ad

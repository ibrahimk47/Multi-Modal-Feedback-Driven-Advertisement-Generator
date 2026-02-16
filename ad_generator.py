def generate_ad(text_sentiment,
                text_emotion,
                user_feedback,
                image_emotion=None,
                detected_object=None):

    if text_sentiment.upper() == "POSITIVE":
        strategy = "Premium Upselling Strategy"
        action_line = "Explore our exclusive premium products."
    elif text_sentiment.upper() == "NEGATIVE":
        strategy = "Recovery & Discount Strategy"
        action_line = "Enjoy special discounts and improved service offers."
    else:
        strategy = "Engagement Strategy"
        action_line = "Discover personalized recommendations curated for you."

    emotion_line = f"Our AI detected that you are feeling '{text_emotion}'."

    if image_emotion:
        emotion_line += f" Visual analysis also detected '{image_emotion}'."

    object_line = ""
    if detected_object:
        object_line = f"You showed interest in '{detected_object}', so we recommend related options."

    advertisement = f"""
🎯 Strategy: {strategy}

"{user_feedback}"

{emotion_line}

{object_line}

✨ {action_line}
"""

    return advertisement

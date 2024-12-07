import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import joblib

# Load pre-trained model
pipe_lr = joblib.load("text_emotion.pkl")

# Updated Dictionary for Mapping Emotions to Emojis
emotions_emoji_dict = {
    "happy": "😂", 
    "sad": "😔", 
    "fear": "😨", 
    "anger": "😠", 
    "surprise": "😮", 
    "neutral": "😐", 
    "disgust": "🤢", 
    "shame": "😳"
}

# Function to Fetch Tweet Text Using Requests
def fetch_tweet_text(tweet_url):
    try:
        # Extract tweet ID from URL
        if "x.com" in tweet_url or "twitter.com" in tweet_url:
            tweet_id = tweet_url.split("/")[-1].split("?")[0]
        else:
            raise ValueError("Invalid URL format. Please enter a valid X.com or Twitter URL.")
        
        # Construct the URL for fetching tweet content
        url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token=a"
        
        # Make the request to fetch the tweet data
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            return data["text"]  # Return the tweet text
        else:
            raise ValueError("Failed to fetch tweet data.")
    except Exception as e:
        st.error(f"Error fetching tweet: {e}")
        return None

# Emotion Prediction Functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Streamlit Application
def main():
    st.title("Tweet Emotion Detection")
    st.subheader("Analyze Emotions in a Tweet")

    with st.form(key='tweet_form'):
        tweet_url = st.text_input("Enter Tweet URL")
        submit_tweet = st.form_submit_button(label='Analyze')

    if submit_tweet:
        # Fetch Tweet Text
        tweet_text = fetch_tweet_text(tweet_url)

        if tweet_text:
            st.success("Fetched Tweet Text")
            st.write(tweet_text)

            # Predict Emotion
            prediction = predict_emotions(tweet_text)
            probability = get_prediction_proba(tweet_text)

            # Display Results
            col1, col2 = st.columns(2)

            with col1:
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.write(f"{prediction} {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions',
                    y='probability',
                    color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()

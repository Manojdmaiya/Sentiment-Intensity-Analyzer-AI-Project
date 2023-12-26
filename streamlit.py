import streamlit as st
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import cv2

positive = cv2.imread("happy_emoji.jpg", cv2.IMREAD_ANYCOLOR)
negative = cv2.imread("sad_emoji.jpeg", cv2.IMREAD_ANYCOLOR)
neutral = cv2.imread("neutral_emoji.png", cv2.IMREAD_ANYCOLOR)

def preprocess_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    tokenized_words = word_tokenize(cleaned_text, "english")

    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)
    return final_words

def analyze_sentiment(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        return "Negative Sentiment", negative
    elif pos > neg:
        return "Positive Sentiment", positive
    else:
        return "Neutral Vibe", neutral

st.title("Sentiment Analysis Web App")

# Add file upload feature for text
uploaded_file = st.file_uploader("Upload Text File", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.text("Original Text:")
    st.text(text)

    cleaned_words = preprocess_text(text)

    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in cleaned_words:
                emotion_list.append(emotion)

    w = Counter(emotion_list)

    sentiment, emoji = analyze_sentiment(text)

    st.text("Emotion List:")
    st.text(emotion_list)

    st.text("Emotion Counter:")
    st.text(dict(w))

    st.text("Sentiment Analysis:")
    st.text(sentiment)

    st.image(emoji, use_column_width=True)

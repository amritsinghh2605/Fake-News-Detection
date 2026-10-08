import streamlit as st
import joblib

# Load model and vectorizer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl"))
# Page title
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)

st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

# User input
news_text = st.text_area(
    "Enter News Text:",
    height=250,
    placeholder="Paste the news article here..."
)

# Prediction
if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        text_vector = vectorizer.transform([news_text])
        prediction = model.predict(text_vector)[0]

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
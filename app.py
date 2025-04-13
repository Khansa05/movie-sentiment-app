import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and find out if it's **Positive** or **Negative**!")

# User input
review = st.text_area("📝 Type your movie review here:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        clean_review = preprocess(review)
        review_vector = vectorizer.transform([clean_review])
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("✅ Sentiment: **Positive** 🎉")
        else:
            st.error("❌ Sentiment: **Negative** 😢")

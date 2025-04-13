## 🎬 Movie Sentiment Analyzer

A simple sentiment analysis web application that classifies IMDb movie reviews as **positive** or **negative** using a Support Vector Machine (SVM) model.

---

## 📌 Project Overview

This project was built as a short-term machine learning task to demonstrate the implementation of a complete NLP pipeline—from dataset preprocessing and model training to web deployment.

---

## 🚀 Live App on Hugging Face

👉 **Try it now**: [Movie Sentiment Analyzer](https://khansaaqureshi-moviesentimentsvm.hf.space)

---

## 🛠️ Features

- Classifies movie reviews using an SVM model
- TF-IDF Vectorization for converting text to numerical features
- Streamlit-powered web interface
- Deployed on Hugging Face Spaces
- Displays predicted sentiment (Positive/Negative)

---

## 📂 Project Structure

- app.py # Streamlit web app 
- svm_model.pkl # Trained SVM model 
- vectorizer.pkl # TF-IDF vectorizer 
- requirements.txt # Python dependencies 
- Sentiment_Analysis_Report.docx # Project report document 
- README.md # Project documentation

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Hugging Face Spaces (deployment)

---

## 💻 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Khansa05/movie-sentiment-app.git
   cd movie-sentiment-app
2. Install requirements:
pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py


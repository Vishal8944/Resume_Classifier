import streamlit as st
import pandas as pd
import re
import nltk
import pickle
import pdfplumber
import docx
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Text cleaning ---
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# --- Resume text extraction ---
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return ""

# --- Load model, vectorizer, and label encoder ---
with open("tfidf.pkl", "rb") as f:  # Loading the TF-IDF vectorizer
    cv = pickle.load(f)

with open("clf.pkl", "rb") as f:  # Loading the trained Random Forest model
    rf = pickle.load(f)

with open("encoder.pkl", "rb") as f:  # Loading the label encoder
    le = pickle.load(f)

# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = clean_text(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = cv.transform([cleaned_text])
    
    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = rf.predict(vectorized_text)

    # Get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 AI Resume Screener")
st.write("Upload a resume (PDF, DOCX, or TXT) to predict its job category.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

    if st.button("Predict Category"):
        if not resume_text.strip():
            st.warning("Couldn't extract any text. Please upload a valid resume file.")
        else:
            category_name = pred(resume_text)
            st.success(f"✅ Predicted Resume Category: **{category_name}**")

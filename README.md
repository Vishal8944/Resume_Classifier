

# 🧠 Resume Classification App

A Streamlit-based web application that uses Natural Language Processing (NLP) and Machine Learning to automatically classify resumes into job categories. This project helps recruiters and HR professionals quickly screen large volumes of resumes and identify the most relevant candidates based on their resume content.

---

## 🔍 Overview

This project builds a resume screening application using a **One-vs-Rest (OVR)** approach with a **Random Forest Classifier**, trained on categorized resume text data. The app predicts job domains such as **Data Science**, **HR**, **Web Development**, and more.

Users can upload resumes in PDF, DOCX, or TXT format. The app extracts the text, preprocesses it using NLP techniques, and predicts the job category using a trained ML model.

---

## 🧠 One-vs-Rest (OVR) Classification Strategy

In this multi-class classification problem, the model uses the **One-vs-Rest (OvR)** strategy:

- For *N* classes, *N* binary classifiers are trained.
- Each classifier learns to distinguish one class from the rest.
- The class with the highest confidence score is selected as the final prediction.

This approach ensures that the model is scalable, interpretable, and robust, especially in scenarios where class imbalance might be present.

---

## 📁 Project Structure

```bash
├── resumeapp.py                 # Streamlit app source code
├── Resume_Screening.ipynb      # Jupyter notebook for data exploration, preprocessing, model training
├── clf.pkl                      # Trained Random Forest model
├── tfidf.pkl                    # TF-IDF vectorizer
├── encoder.pkl                  # Label encoder for category mapping
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

---

## 🚀 Features

- 📄 Upload resumes in PDF, DOCX, or TXT format
- ✨ Clean and preprocess resume text using NLP
- 🔍 Predict job category using a trained Random Forest (OVR) model
- ⚙️ TF-IDF vectorization for feature extraction
- 🧑‍💻 Simple and interactive UI built with Streamlit

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** - UI framework
- **Scikit-learn** - ML algorithms
- **PDFPlumber / python-docx** - File handling
- **NLTK** - Text preprocessing (stopwords)
- **TF-IDF** - Feature extraction
- **One-vs-Rest (OVR)** - Classification strategy for multi-label prediction

---

## 🧪 How it Works

1. **Upload Resume**: User uploads a resume file (PDF/DOCX/TXT).
2. **Text Extraction**: Text is extracted using libraries (`pdfplumber`, `docx`, etc.).
3. **Preprocessing**: Text is cleaned (lowercased, special characters removed, stopwords removed).
4. **Vectorization**: Cleaned text is converted to numerical features via TF-IDF.
5. **Classification**: Random Forest model (using OVR) predicts the most likely job category.
6. **Result Display**: The predicted job category is shown in the app.

---

## 🧑‍💻 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/resume-classifier-app.git
cd resume-classifier-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run resumeapp.py
```

---

## 📄 Sample Categories

Here are some of the job categories the model can classify:

- Data Science
- Web Development
- Human Resources
- Operations
- Testing
- DevOps
- Business Analyst
- Others depending on training data

---

## 📦 Requirements

```txt
streamlit
pandas
scikit-learn
nltk
pdfplumber
python-docx
```

---

## 💡 Future Improvements

- 🔍 Add resume-to-job-description matching
- 🧠 Use transformer-based models (e.g., BERT)
- 🌐 Add multi-language support
- 🧾 Enhance UI with analytics dashboard for HR

---

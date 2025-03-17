# ATS Resume Ranking System Project - https://resume-score-checker.streamlit.app/

🚀 **ATS Resume Ranking System**  
An AI-powered web application that automates the process of ranking resumes for Data Science Fresher roles. It uses Natural Language Processing (NLP) techniques to extract and preprocess resume text, and applies TF-IDF Vectorization with Cosine Similarity to calculate an ATS (Applicant Tracking System) score for each resume.

## 📝 Features
- Upload multiple resumes (PDF supported).
- Enter or paste a job description.
- Preprocesses resumes and job descriptions with tokenization, stopword removal, and lemmatization.
- Computes similarity scores between resumes and job descriptions.
- Ranks resumes based on ATS scores (cosine similarity).
- Clean, responsive dark-themed UI using Streamlit.

## 🔧 Technologies Used
- Python
- Streamlit (for UI)
- NLP (SpaCy, NLTK)
- PDF Processing (pdfplumber)
- Machine Learning (TF-IDF Vectorizer, Cosine Similarity from scikit-learn)

## 📂 How to Run
1. Clone the repo  
   `git clone https://github.com/ujjwalmisra/ATS_project.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Run the app  
   `streamlit run app.py`

## ✨ Future Improvements
- Support for DOCX files.
- Advanced machine learning models for better matching.
- User authentication and profile saving.
- Dashboard with analytics and insights.

## 👨‍💻 Author
- [Your Name](https://github.com/ujjwalmisra)

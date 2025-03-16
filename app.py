import streamlit as st
import os
import pdfplumber
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------- SETUP ---------------- #

# Download NLTK stopwords (only once, skip next time)
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Create "resumes" folder if it doesn't exist
if not os.path.exists("resumes"):
    os.makedirs("resumes")

# ----------- FUNCTIONS --------------- #

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphanumeric
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    
    # Lemmatize using SpaCy
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized_tokens)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Some pages may return None
                text += page_text
    return text

def calculate_ats_score(jd, resumes_text):
    # Combine JD and all resumes into a single corpus
    corpus = [jd] + resumes_text  # JD at index 0, resumes from 1 onwards
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarity (JD vs Resumes)
    jd_vector = vectors[0:1]
    resume_vectors = vectors[1:]
    
    similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    
    # You can return the similarity scores as ATS scores
    ats_scores = similarity_scores * 100  # Convert to percentage
    return ats_scores

# --------- STREAMLIT APP ------------- #

st.title("ATS Resume Ranking System (Data Science Fresher Job)")

# Upload resumes
resume_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf"], accept_multiple_files=True)

# Input Job Description (you can edit this later)
job_description_input = st.text_area("Enter Job Description", value=""" 
Fresher data science jobs typically require a strong foundation in programming languages like Python or R, and experience with machine learning frameworks and libraries such as TensorFlow, PyTorch, Scikit-learn, and XGBoost. Candidates should have a strong understanding of statistical methods, data analysis, and machine learning algorithms. Additionally, proficiency in data manipulation tools like Pandas, NumPy, and SQL, and familiarity with big data technologies (e.g., Hadoop, Spark) and cloud platforms (e.g., AWS, GCP, Azure) are highly desirable. Experience with version control systems like Git and knowledge of deep learning, natural language processing (NLP), or computer vision are also beneficial. These roles often involve collecting, cleaning, analyzing, and interpreting data to solve complex problems and make data-driven decisions. Data scientists in fresher roles may also be involved in developing predictive models, improving data quality, and enhancing data processing efficiency. Communication skills are crucial, as data scientists must clearly present complex data insights through visualizations and written reports and have strong interpersonal skills for effective collaboration with team members and stakeholders.
""")

# Click to Process Resumes
if st.button("Process Resumes"):
    if not resume_files:
        st.warning("Please upload at least one resume file.")
    else:
        resume_texts = []
        resume_names = []
        
        # Loop through uploaded resumes
        for resume in resume_files:
            resume_path = os.path.join("resumes", resume.name)
            
            # Save file locally
            with open(resume_path, "wb") as f:
                f.write(resume.read())
            
            # Extract text
            if resume.type == "application/pdf":
                text = extract_text_from_pdf(resume_path)
            else:
                text = ""
                st.warning(f"{resume.name} is not a supported format in this version.")
            
            # Preprocess text and save for ATS score calculation
            preprocessed_text = preprocess_text(text)
            resume_texts.append(preprocessed_text)
            resume_names.append(resume.name)
            
            # Optional: Show extracted text
            st.subheader(f"Extracted Text from {resume.name}")
            st.write(text[:500] + " ...")  # Show first 500 characters
        
        # Preprocess Job Description
        preprocessed_jd = preprocess_text(job_description_input)
        
        # Calculate ATS Scores
        ats_scores = calculate_ats_score(preprocessed_jd, resume_texts)
        
        # Show Results
        st.header("ATS Scores & Ranking")
        ranking = sorted(zip(resume_names, ats_scores), key=lambda x: x[1], reverse=True)
        
        for idx, (name, score) in enumerate(ranking, start=1):
            st.write(f"{idx}. {name} - ATS Score: {score:.2f}%")

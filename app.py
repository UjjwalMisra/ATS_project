import streamlit as st
import os
import re
import pandas as pd
import pdfplumber
import docx2txt
import spacy
from spacy.matcher import PhraseMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO

# -------------- SETUP ---------------- #
st.set_page_config(page_title="Data Science ATS Resume Ranker", page_icon="📊", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #F8FAFC;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.2rem;
        border-radius: 2px;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Download necessary resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

# Load SpaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

download_nltk_resources()
nlp = load_spacy_model()

# Create folders if they don't exist
if not os.path.exists("resumes"):
    os.makedirs("resumes")

# ----------- FUNCTIONS --------------- #

def extract_text_from_pdf(file_path):
    """Extract text from PDF files with better handling of formatting."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX files."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def preprocess_text(text):
    """Preprocess text with improved cleaning and normalization."""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize using SpaCy
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized_tokens)

def extract_skills(text, skills_list):
    """Extract skills from text using SpaCy's PhraseMatcher."""
    skills_found = []
    
    # Create a pattern for each skill
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill.lower()) for skill in skills_list]
    matcher.add("SKILLS", None, *patterns)
    
    # Process the text and find matches
    doc = nlp(text.lower())
    matches = matcher(doc)
    
    for match_id, start, end in matches:
        span = doc[start:end]
        skills_found.append(span.text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_skills = [skill for skill in skills_found if not (skill in seen or seen.add(skill))]
    
    return unique_skills

def extract_email(text):
    """Extract email address from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not found"

def extract_phone(text):
    """Extract phone number from text."""
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else "Not found"

def calculate_ats_score(jd, resume_text, skills_list):
    """Calculate ATS score using multiple factors."""
    # Preprocess texts
    preprocessed_jd = preprocess_text(jd)
    preprocessed_resume = preprocess_text(resume_text)
    
    # 1. Calculate TF-IDF similarity (50% of score)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocessed_jd, preprocessed_resume])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    tfidf_score = cosine_sim * 50
    
    # 2. Skills matching (30% of score)
    jd_skills = extract_skills(jd, skills_list)
    resume_skills = extract_skills(resume_text, skills_list)
    
    if jd_skills:
        skills_match_pct = len(set(resume_skills) & set(jd_skills)) / len(set(jd_skills)) * 100
    else:
        skills_match_pct = 0
        
    skills_score = skills_match_pct * 0.3
    
    # 3. Keyword frequency analysis (20% of score)
    # Extract important keywords from JD (excluding skills already counted)
    jd_doc = nlp(jd)
    keywords = [token.text.lower() for token in jd_doc if token.pos_ in ('NOUN', 'PROPN') 
                and token.text.lower() not in [skill.lower() for skill in skills_list]
                and len(token.text) > 3 and token.text.lower() not in stopwords.words('english')]
    
    # Count how many of these keywords appear in resume
    resume_lower = resume_text.lower()
    keyword_matches = sum(1 for keyword in set(keywords) if keyword in resume_lower)
    
    if keywords:
        keyword_score = (keyword_matches / len(set(keywords))) * 20
    else:
        keyword_score = 0
    
    # Calculate total score
    total_score = tfidf_score + skills_score + keyword_score
    
    # Return components for detailed analysis
    return {
        "total_score": total_score,
        "tfidf_score": tfidf_score,
        "skills_score": skills_score,
        "keyword_score": keyword_score,
        "matched_skills": list(set(resume_skills) & set(jd_skills)),
        "missing_skills": list(set(jd_skills) - set(resume_skills)),
        "all_skills": resume_skills
    }

def generate_word_cloud(text):
    """Generate a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=150, contour_width=3, contour_color='steelblue').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def get_download_link(df):
    """Generate a download link for the results dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resume_rankings.csv">Download Results as CSV</a>'
    return href

def highlight_keywords(text, keywords):
    """Highlight keywords in text for display."""
    highlighted_text = text
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span class="highlight">{keyword}</span>', highlighted_text)
    return highlighted_text

# --------- STREAMLIT APP ------------- #

st.markdown('<h1 class="main-header">Advanced ATS Resume Ranking System</h1>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📋 Job Description & Upload", "📊 Rankings & Analysis", "👁️ Resume Viewer", "ℹ️ About"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Job Description</h2>', unsafe_allow_html=True)
    
    # Sample Job Description
    default_jd = """
    Fresher Data Scientist Position
    
    Requirements:
    - Bachelor's degree in Computer Science, Statistics, Mathematics, or related field
    - Strong programming skills in Python or R
    - Experience with machine learning frameworks (TensorFlow, PyTorch, Scikit-learn)
    - Knowledge of data manipulation libraries (Pandas, NumPy)
    - Understanding of statistical methods and machine learning algorithms
    - Familiarity with SQL and database concepts
    - Experience with data visualization tools (Matplotlib, Seaborn, Plotly)
    - Basic understanding of deep learning concepts
    - Excellent problem-solving and communication skills
    - Bonus skills: Natural Language Processing, Computer Vision, Cloud platforms (AWS/GCP/Azure)
    
    Responsibilities:
    - Collect, clean, and preprocess data
    - Develop and implement machine learning models
    - Analyze complex datasets to extract meaningful insights
    - Create data visualizations to communicate findings
    - Collaborate with cross-functional teams on data-driven solutions
    - Stay updated with the latest developments in data science
    """
    
    # Allow user to edit or use their own job description
    job_description = st.text_area("Enter or edit the job description", value=default_jd, height=300)
    
    # Pre-defined skills for data science positions
    default_skills = [
        "Python", "R", "SQL", "Pandas", "NumPy", "SciPy", "Scikit-learn", "TensorFlow", 
        "PyTorch", "Keras", "NLTK", "spaCy", "Matplotlib", "Seaborn", "Plotly", "Tableau",
        "Power BI", "Excel", "Statistics", "Machine Learning", "Deep Learning", "NLP",
        "Computer Vision", "ETL", "Data Mining", "Big Data", "Hadoop", "Spark", "AWS",
        "GCP", "Azure", "Docker", "Kubernetes", "Git", "A/B Testing", "Time Series Analysis"
    ]
    
    # Allow users to edit the skills list
    skills_input = st.text_area("Data Science Skills (one per line, edit as needed)", 
                               value="\n".join(default_skills), height=200)
    skills_list = [skill.strip() for skill in skills_input.split("\n") if skill.strip()]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Upload Resumes</h2>', unsafe_allow_html=True)
    
    # Upload resumes
    resume_files = st.file_uploader(
        "Upload Resumes (PDF or DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True,
        help="Select multiple resume files to analyze them against the job description"
    )
    
    process_button = st.button("Process Resumes", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state for storing results
if 'resume_results' not in st.session_state:
    st.session_state.resume_results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_resume_index' not in st.session_state:
    st.session_state.selected_resume_index = 0

# Process resumes when button is clicked
if process_button and resume_files:
    with st.spinner('Processing resumes... This may take a moment'):
        results = []
        
        for resume in resume_files:
            # Save file temporarily
            resume_path = os.path.join("resumes", resume.name)
            with open(resume_path, "wb") as f:
                f.write(resume.read())
            
            # Extract text based on file type
            if resume.name.endswith('.pdf'):
                resume_text = extract_text_from_pdf(resume_path)
            elif resume.name.endswith('.docx'):
                resume_text = extract_text_from_docx(resume_path)
            else:
                st.warning(f"{resume.name} has an unsupported format.")
                continue
            
            # Calculate scores
            scores = calculate_ats_score(job_description, resume_text, skills_list)
            
            # Extract contact info
            email = extract_email(resume_text)
            phone = extract_phone(resume_text)
            
            # Store results
            results.append({
                "file_name": resume.name,
                "text": resume_text,
                "total_score": scores["total_score"],
                "tfidf_score": scores["tfidf_score"],
                "skills_score": scores["skills_score"], 
                "keyword_score": scores["keyword_score"],
                "matched_skills": scores["matched_skills"],
                "missing_skills": scores["missing_skills"],
                "all_skills": scores["all_skills"],
                "email": email,
                "phone": phone
            })
        
        # Sort results by total score
        results.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Store in session state
        st.session_state.resume_results = results
        st.session_state.analysis_complete = True
        
        # Success message
        if results:
            st.success(f"Successfully processed {len(results)} resumes!")
        else:
            st.error("No valid resumes were processed. Please check the file formats.")

# Display results in rankings tab
with tab2:
    if st.session_state.analysis_complete and st.session_state.resume_results:
        st.markdown('<h2 class="section-header">Resume Rankings</h2>', unsafe_allow_html=True)
        
        results = st.session_state.resume_results
        
        # Create DataFrame for display
        df_display = pd.DataFrame({
            "Rank": range(1, len(results) + 1),
            "Resume": [r["file_name"] for r in results],
            "ATS Score": [f"{r['total_score']:.1f}%" for r in results],
            "Content Match": [f"{r['tfidf_score']:.1f}%" for r in results],
            "Skills Match": [f"{r['skills_score']:.1f}%" for r in results],
            "Keyword Match": [f"{r['keyword_score']:.1f}%" for r in results],
            "Matched Skills": [", ".join(r["matched_skills"][:5]) + ("..." if len(r["matched_skills"]) > 5 else "") for r in results],
        })
        
        st.dataframe(df_display, use_container_width=True)
        
        # Download results button
        df_export = pd.DataFrame({
            "Rank": range(1, len(results) + 1),
            "Resume": [r["file_name"] for r in results],
            "ATS_Score": [r['total_score'] for r in results],
            "Content_Match": [r['tfidf_score'] for r in results],
            "Skills_Match": [r['skills_score'] for r in results],
            "Keyword_Match": [r['keyword_score'] for r in results],
            "Matched_Skills": [", ".join(r["matched_skills"]) for r in results],
            "Missing_Skills": [", ".join(r["missing_skills"]) for r in results],
            "Email": [r["email"] for r in results],
            "Phone": [r["phone"] for r in results]
        })
        
        st.markdown(get_download_link(df_export), unsafe_allow_html=True)
        
        # Visualization section
        st.markdown('<h2 class="section-header">Visualizations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of top rankings
            fig = px.bar(
                df_display.head(min(10, len(results))),
                x="Resume",
                y=[s.rstrip("%") for s in df_display["ATS Score"].head(min(10, len(results)))],
                title="Top Resume Rankings",
                labels={"y": "ATS Score (%)", "Resume": ""},
                color=[float(s.rstrip("%")) for s in df_display["ATS Score"].head(min(10, len(results)))],
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart for top 5 resumes
            categories = ['Content Match', 'Skills Match', 'Keyword Match']
            
            fig = go.Figure()
            
            for i, result in enumerate(results[:min(5, len(results))]):
                fig.add_trace(go.Scatterpolar(
                    r=[result['tfidf_score'], result['skills_score'], result['keyword_score']],
                    theta=categories,
                    fill='toself',
                    name=f"{i+1}. {result['file_name'][:15]}..."
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 50]
                    )),
                showlegend=True,
                title="Score Components Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills analysis
        st.markdown('<h2 class="section-header">Skills Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most common matched skills across all resumes
            all_matched_skills = [skill for r in results for skill in r["matched_skills"]]
            skill_counts = {}
            for skill in all_matched_skills:
                if skill in skill_counts:
                    skill_counts[skill] += 1
                else:
                    skill_counts[skill] = 1
            
            skill_df = pd.DataFrame({
                "Skill": list(skill_counts.keys()),
                "Count": list(skill_counts.values())
            }).sort_values("Count", ascending=False).head(15)
            
            if not skill_df.empty:
                fig = px.bar(
                    skill_df,
                    x="Count",
                    y="Skill",
                    title="Most Common Skills in Top Resumes",
                    orientation='h',
                    color="Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No common skills found across resumes.")
        
        with col2:
            # Most common missing skills across all resumes
            all_missing_skills = [skill for r in results for skill in r["missing_skills"]]
            missing_skill_counts = {}
            for skill in all_missing_skills:
                if skill in missing_skill_counts:
                    missing_skill_counts[skill] += 1
                else:
                    missing_skill_counts[skill] = 1
            
            missing_skill_df = pd.DataFrame({
                "Skill": list(missing_skill_counts.keys()),
                "Count": list(missing_skill_counts.values())
            }).sort_values("Count", ascending=False).head(15)
            
            if not missing_skill_df.empty:
                fig = px.bar(
                    missing_skill_df,
                    x="Count",
                    y="Skill",
                    title="Most Commonly Missing Skills",
                    orientation='h',
                    color="Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No common missing skills found across resumes.")
    else:
        st.info("Upload and process resumes to see rankings and analysis here.")

# Resume viewer tab
with tab3:
    if st.session_state.analysis_complete and st.session_state.resume_results:
        st.markdown('<h2 class="section-header">Resume Content Viewer</h2>', unsafe_allow_html=True)
        
        results = st.session_state.resume_results
        
        # Select resume to view
        resume_options = [f"{i+1}. {r['file_name']} (Score: {r['total_score']:.1f}%)" 
                         for i, r in enumerate(results)]
        
        selected_resume = st.selectbox(
            "Select a resume to view",
            options=resume_options,
            index=st.session_state.selected_resume_index
        )
        
        # Update selected index in session state
        selected_index = resume_options.index(selected_resume)
        st.session_state.selected_resume_index = selected_index
        
        # Display resume details
        result = results[selected_index]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h3>Resume Content</h3>", unsafe_allow_html=True)
            
            # Highlight matched skills in the resume text
            highlighted_text = result["text"]
            for skill in result["matched_skills"]:
                highlighted_text = re.sub(
                    r'\b' + re.escape(skill) + r'\b', 
                    f'<span style="background-color: #BBDEFB; padding: 0px 3px; border-radius: 3px;">{skill}</span>',
                    highlighted_text,
                    flags=re.IGNORECASE
                )
            
            st.markdown(f'<div style="max-height: 500px; overflow-y: auto; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap;">{highlighted_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>Resume Analytics</h3>", unsafe_allow_html=True)
            
            # Display scores with progress bars
            st.markdown("<b>ATS Score Components:</b>", unsafe_allow_html=True)
            st.progress(result["tfidf_score"]/100)
            st.markdown(f"Content Match: {result['tfidf_score']:.1f}%", unsafe_allow_html=True)
            
            st.progress(result["skills_score"]/100)
            st.markdown(f"Skills Match: {result['skills_score']:.1f}%", unsafe_allow_html=True)
            
            st.progress(result["keyword_score"]/100)
            st.markdown(f"Keyword Match: {result['keyword_score']:.1f}%", unsafe_allow_html=True)
            
            st.progress(result["total_score"]/100)
            st.markdown(f"<b>Total ATS Score: {result['total_score']:.1f}%</b>", unsafe_allow_html=True)
            
            st.markdown("<b>Contact Information:</b>", unsafe_allow_html=True)
            st.markdown(f"Email: {result['email']}", unsafe_allow_html=True)
            st.markdown(f"Phone: {result['phone']}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Matched skills
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Skills Analysis</h4>", unsafe_allow_html=True)
            
            st.markdown("<b>Matched Skills:</b>", unsafe_allow_html=True)
            matched_skills_str = ", ".join(result["matched_skills"])
            st.markdown(f'<div style="color: green;">{matched_skills_str}</div>', unsafe_allow_html=True)
            
            st.markdown("<b>Missing Skills:</b>", unsafe_allow_html=True)
            missing_skills_str = ", ".join(result["missing_skills"])
            st.markdown(f'<div style="color: red;">{missing_skills_str}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Improvement suggestions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Improvement Suggestions</h4>", unsafe_allow_html=True)
            
            suggestions = []
            
            if result["missing_skills"]:
                suggestions.append(f"• Add missing skills: {', '.join(result['missing_skills'][:3])}" + 
                                 ("..." if len(result['missing_skills']) > 3 else ""))
            
            if result["tfidf_score"] < 25:
                suggestions.append("• Improve content alignment with job description")
                
            if result["keyword_score"] < 10:
                suggestions.append("• Include more relevant keywords from the job description")
                
            if not suggestions:
                suggestions.append("• Resume is well-aligned with job requirements")
                
            for suggestion in suggestions:
                st.markdown(suggestion, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Word cloud visualization
        st.markdown('<h3 class="section-header">Resume Word Cloud</h3>', unsafe_allow_html=True)
        wc_fig = generate_word_cloud(result["text"])
        st.pyplot(wc_fig)
    else:
        st.info("Upload and process resumes to view their contents here.")

# About tab
with tab4:
    st.markdown('<h2 class="section-header">About this ATS Resume Ranking System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This advanced ATS (Applicant Tracking System) Resume Ranking tool helps evaluate resumes for data science positions using natural language processing and machine learning techniques.
    
    ### How It Works
    
    The system uses a multi-faceted approach to score resumes:
    
    1. **Content Matching (50%)**: Uses TF-IDF vectorization and cosine similarity to compare the overall content of resumes against the job description.
    
    2. **Skills Matching (30%)**: Identifies specific data science skills in both the job description and resumes, then calculates a matching percentage.
    
    3. **Keyword Analysis (20%)**: Extracts important keywords (excluding skills) from the job description and checks for their presence in resumes.
    
    ### Features
    
    - **Multi-format support**: Handles PDF and DOCX resume formats
    - **Interactive visualizations**: Compare rankings, score breakdowns, and skill distributions
    - **Detailed analysis**: View matched/missing skills and improvement suggestions
    - **Resume highlighting**: Automatically highlights matched skills in resume text
    - **Export functionality**: Download complete results for offline analysis
    
    ### Tips for Candidates
    
    - Include relevant skills explicitly mentioned in the job description
    - Use industry-standard terminology for data science concepts and tools
    - Ensure your resume contains keywords from the job description
    - Tailor your resume to specifically address the job requirements
    
    ### Data Privacy
    
    Resumes are processed locally within this application and are not stored permanently or sent to external services.
    """)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd; color: #666;">
    <p>Advanced ATS Resume Ranking System for Data Science Positions | © 2023</p>
</div>
""", unsafe_allow_html=True)

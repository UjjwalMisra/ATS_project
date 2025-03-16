import streamlit as st
st.header("Upload Resumes")
resume_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

import os

# Make sure the resumes folder exists
if not os.path.exists("resumes"):
    os.makedirs("resumes")

# Save each uploaded resume file
for resume in resume_files:
    with open(os.path.join("resumes", resume.name), "wb") as f:
        f.write(resume.read())

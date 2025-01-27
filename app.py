from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
from PIL import Image
import pdf2image
import pytesseract
import re
import google.generativeai as genai

# Configure Google API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini response based on job description and resume
def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

# Function to set up PDF to image conversion and extract text using OCR
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Path to the Poppler binaries
        poppler_path = r"C:\Program Files (x86)\poppler\Library\bin"

        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)

        # Use OCR (pytesseract) to extract text from the first page image
        first_page = images[0]
        extracted_text = pytesseract.image_to_string(first_page)

        return extracted_text
    else:
        raise FileNotFoundError("No file uploaded")

# Function to extract relevant skills from text (Job Description / Resume)
def extract_skills(text):
    # Define common skills to look for (can be expanded as needed)
    skill_keywords = [
        'Python', 'Data Science', 'Machine Learning', 'Deep Learning', 'AI', 'SQL',
        'Java', 'JavaScript', 'C++', 'Cloud Computing', 'AWS', 'Azure', 'Docker',
        'Big Data', 'Data Engineering', 'Project Management', 'Agile', 'Excel', 'Tableau'
    ]
    
    skills_found = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills_found.append(skill)
    
    return skills_found

# Function to suggest learning resources based on missing skills
def get_learning_resources(missing_skills):
    learning_resources = {
        'Python': "https://www.learnpython.org/",
        'Data Science': "https://www.coursera.org/specializations/jhu-data-science",
        'Machine Learning': "https://www.coursera.org/learn/machine-learning",
        'Deep Learning': "https://www.deeplearning.ai/online-deep-learning-courses/",
        'AI/ML': "https://www.edx.org/professional-certificate/harvardx-data-science-ai",
        'SQL': "https://www.codecademy.com/learn/learn-sql"
    }

    resources = {}
    for skill in missing_skills:
        if skill in learning_resources:
            resources[skill] = learning_resources[skill]
    return resources

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")

input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are an ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. 
Give me the percentage match if the resume matches the job description. First, the output should come as a percentage and then keywords missing, followed by final thoughts.
"""

if submit1:
    if uploaded_file is not None:
        # Extract text from uploaded PDF using OCR
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)

        # Extract skills from resume text
        resume_skills = extract_skills(pdf_content)
        st.subheader("Skills Detected in Resume")
        st.write(resume_skills)

        # Get missing skills and suggest learning resources
        missing_skills = ["Python", "Data Science", "Machine Learning"]  # Example missing skills; could be dynamically extracted
        learning_resources = get_learning_resources(missing_skills)
        if learning_resources:
            st.subheader("Suggested Learning Resources:")
            for skill, link in learning_resources.items():
                st.write(f"- **{skill}**: [Learn More]({link})")

    else:
        st.write("Please upload the resume")

elif submit3:
    if uploaded_file is not None:
        # Extract text from uploaded PDF using OCR
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)

        # Extract skills from resume text
        resume_skills = extract_skills(pdf_content)
        st.subheader("Skills Detected in Resume")
        st.write(resume_skills)

        # Get missing skills and suggest learning resources
        missing_skills = ["Python", "SQL", "AI/ML"]  # Example missing skills; could be dynamically extracted
        learning_resources = get_learning_resources(missing_skills)
        if learning_resources:
            st.subheader("Suggested Learning Resources:")
            for skill, link in learning_resources.items():
                st.write(f"- **{skill}**: [Learn More]({link})")

    else:
        st.write("Please upload the resume")

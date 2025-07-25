# app.py

import streamlit as st
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
import fitz  # PyMuPDF
from streamlit_option_menu import option_menu
from fpdf import FPDF
import logging

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(
    page_title="Recruit Nepal CV Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Environment and Configure API ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("🔴 Google API Key not found! Please set it in your Streamlit secrets or a local .env file.")
        st.stop()
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"🔴 Error configuring Google API: {e}")
    st.stop()

# --- Gemini API & PDF Functions ---

def get_gemini_response(prompt, return_raw_text=False):
    """Sends a prompt to the Gemini API. Returns cleaned JSON by default."""
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    try:
        response = model.generate_content(prompt)
        text = response.text
        if return_raw_text:
            return text

        cleaned_text = re.sub(r'```json\s*|\s*```', '', text.strip())
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        st.warning("Could not find a valid JSON object in the AI response. Please try again.")
        logging.warning(f"No JSON object found in response: {cleaned_text}")
        return None
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        st.error(f"An error occurred while communicating with the AI model: {e}")
        return None

def get_pdf_text(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        file_bytes = uploaded_file.getvalue()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def create_resume_pdf(text_content):
    """Generates a PDF from a string of text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=10)
    for line in text_content.split('\n'):
        if line.isupper() and len(line.split()) < 5 and len(line.strip()) > 1:
            pdf.set_font("Helvetica", 'B', 12)
            pdf.ln(5)
            encoded_line = line.strip().encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, encoded_line)
            pdf.ln(2)
            pdf.set_font("Helvetica", '', 10)
        else:
            encoded_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, encoded_line)
    return pdf.output(dest='S').encode('latin-1')


# --- Prompt Definitions ---

COMPREHENSIVE_ANALYSIS_PROMPT = """
You are an expert ATS and professional resume coach. Conduct a deep analysis of the provided resume.
**Crucially, you must provide detailed feedback with AT LEAST 5 strengths and AT LEAST 5 areas for improvement.**

Return a SINGLE, VALID JSON object with the exact structure below.

Resume Text:
---
{resume_text}
---

Job Description (if available, otherwise "N/A"):
---
{job_description}
---

**JSON OUTPUT FORMAT:**
{{
  "overall_ats_score": {{ "score": 82, "summary": "Strong foundation, needs optimization." }},
  "job_match_score": {{ "score": 75, "summary": "Good alignment, missing some keywords." }},
  "industry_domain_matching": {{ "detected_domain": "Data Science", "suggested_role": "Data Analyst" }},
  "resume_structure_formatting": {{ "word_count": 489, "section_completeness": {{ "Summary": true, "Experience": true, "Projects": true, "Education": true, "Certifications": false, "Skills": true }} }},
  "keyword_ats_optimization": {{ "top_missing_keywords": ["A/B Testing", "ETL Pipelines", "Predictive Modeling"] }},
  "grammar_clarity_score": {{ "score": 92, "feedback": "Grammar is strong. Found 2 passive sentences." }},
  "repetitive_generic_phrases_score": {{ "score": 78, "detected_phrases": ["team player"], "feedback": "Avoid clichés like 'team player'." }},
  "quantified_impact_score": {{ "score": 65, "feedback": "Only 2 of 8 bullets have metrics." }},
  "detailed_feedback": {{
    "strengths": [
      "The resume has a clean, professional format that is easy to read.",
      "Contact information is complete and easily accessible.",
      "The 'Skills' section is well-organized and lists relevant technologies.",
      "Action verbs are used effectively in the 'Experience' section (e.g., 'Developed', 'Managed').",
      "The education section is concise and correctly placed."
    ],
    "areas_for_improvement": [
      "Quantify achievements more. Instead of 'Improved performance', use 'Improved performance by 15%'.",
      "Tailor the summary to the specific job. Mention the company or role you're applying for.",
      "The resume is missing a 'Projects' section, which is crucial for technical roles to showcase hands-on work.",
      "Some bullet points are too long. Aim for 1-2 lines per bullet for better readability.",
      "Add a link to your LinkedIn profile and GitHub/Portfolio in the contact section."
    ]
  }}
}}
"""

ENHANCED_RESUME_PROMPT_TEMPLATE = """
You are an elite career coach and professional resume writer. Your task is to rewrite and enhance the entire resume text provided below.
- Strengthen language using powerful action verbs.
- Improve clarity, conciseness, and impact.
- Correct any grammatical or spelling errors.
- Ensure a professional tone throughout.
- CRUCIALLY, preserve the original structure and section headings (e.g., "PROFESSIONAL SUMMARY", "EXPERIENCE", "SKILLS").
Do not add any commentary, introductions, or summaries. Return ONLY the full, rewritten resume text.
Original Resume Text:
---
{resume_text}
---
"""

# --- UI Styling ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F5; }
    #MainMenu, footer, header {visibility: hidden;}
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E293B; text-align: center; margin-bottom: 10px; }
    .main-subheader { text-align: center; color: #475569; margin-bottom: 30px; }
    .metric-card {
        background-color: white; border-radius: 12px; padding: 20px;
        border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        text-align: center; height: 100%;
    }
    .metric-card-label { font-size: 1rem; color: #64748B; margin-bottom: 8px; font-weight: 500;}
    .metric-card-value { font-size: 2rem; font-weight: 700; color: #1E293B; }
    .metric-card-feedback { font-size: 0.85rem; color: #475569; margin-top: 10px; }
    .score-color-green { color: #16A34A; }
    .score-color-orange { color: #F97316; }
    .score-color-red { color: #DC2626; }
    
    /* MODIFIED STYLE FOR DASHED BOX */
    .dashed-feedback-box {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        border: 2px dashed #A0AEC0; /* Dashed border */
        height: 100%;
    }
    .feedback-list { list-style-position: inside; padding-left: 5px; }
    .feedback-list li { margin-bottom: 10px; line-height: 1.5; color: #334155; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions for UI ---
def get_score_color(score):
    if score >= 85: return "green"
    if score >= 70: return "orange"
    return "red"

def display_metric_card(label, value, feedback="", color_class=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value {color_class}">{value}</div>
        <p class="metric-card-feedback">{feedback}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
if "enhanced_resume_text" not in st.session_state: st.session_state.enhanced_resume_text = None
if "resume_text" not in st.session_state: st.session_state.resume_text = ""
if "file_name" not in st.session_state: st.session_state.file_name = None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.image("https://via.placeholder.com/200x60.png?text=Recruit+Nepal", use_column_width=True)
    st.title("📄 CV Input")
    st.markdown("Upload your CV and paste a job description for a complete analysis.")
    
    uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])
    job_description = st.text_area("Paste Target Job Description (Optional)", height=200, placeholder="Example: We are looking for a Python developer...")

    analyze_button = st.button("Analyze My Resume", use_container_width=True, type="primary")

# --- Logic to Process Upload and Analysis ---
if uploaded_file:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.resume_text = get_pdf_text(uploaded_file)
        st.session_state.file_name = uploaded_file.name
        st.session_state.analysis_results = None 
        st.session_state.enhanced_resume_text = None

if analyze_button and st.session_state.resume_text:
    st.session_state.analysis_results = None
    st.session_state.enhanced_resume_text = None
    with st.spinner("🤖 AI is performing a deep analysis of your resume... This may take a moment."):
        prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
            resume_text=st.session_state.resume_text,
            job_description=job_description or "N/A"
        )
        response = get_gemini_response(prompt)
        if response:
            try:
                st.session_state.analysis_results = json.loads(response)
                st.success("Analysis Complete! View your dashboard below.")
            except json.JSONDecodeError as e:
                st.error("Error: The AI response was not in the expected format. Please try again.")
                logging.error(f"JSON Decode Error: {e}\nRaw Response:\n{response}")
        else:
            st.error("Failed to get a response from the AI. Please check your API key and try again.")
elif analyze_button and not st.session_state.resume_text:
    st.warning("Please upload a resume before analyzing.")


# --- Main Page UI ---
st.markdown("<div class='main-header'>Recruit Nepal CV Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='main-subheader'>Get instant, AI-powered feedback to land your dream job in Nepal.</div>", unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=["📊 Score Dashboard", "📄 Raw Text Viewer", "✨ AI Editor"],
    icons=["clipboard-data-fill", "file-earmark-text-fill", "magic"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#F0F2F5", "border": "1px solid #E2E8F0", "border-radius": "12px"},
        "icon": {"color": "#475569", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "font-weight": "500", "text-align": "center", "margin":"0px", "--hover-color": "#E2E8F0"},
        "nav-link-selected": {"background-color": "#1E40AF", "color": "white"}
    }
)

# --- Tab Content ---
if selected == "📊 Score Dashboard":
    if st.session_state.analysis_results:
        data = st.session_state.analysis_results
        
        st.subheader("🎯 Top Metrics at a Glance")
        st.markdown("---")
        
        cols = st.columns(3)
        with cols[0]:
            score_data = data.get('overall_ats_score', {})
            score = score_data.get('score', 0)
            color = get_score_color(score)
            display_metric_card("Your ATS Compatibility Score", f"{score}/100", score_data.get('summary', 'N/A'), f"score-color-{color}")

        with cols[1]:
            score_data = data.get('job_match_score', {})
            score = score_data.get('score', 0) if job_description else "N/A"
            value = f"{score}/100" if isinstance(score, int) else "N/A"
            feedback = score_data.get('summary', 'Paste a job description for this score.')
            color = get_score_color(score) if isinstance(score, int) else ""
            display_metric_card("Your Match for This Job", value, feedback, f"score-color-{color}")
            
        with cols[2]:
            score_data = data.get('quantified_impact_score', {})
            score = score_data.get('score', 0)
            color = get_score_color(score)
            display_metric_card("Achievements with Numbers?", f"{score}/100", score_data.get('feedback', 'N/A'), f"score-color-{color}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("✅ Good Points & 💡 Bad Points")
        st.markdown("---")
        feedback_data = data.get('detailed_feedback', {})
        strengths = feedback_data.get('strengths', [])
        improvements = feedback_data.get('areas_for_improvement', [])
        
        col1, col2 = st.columns(2)
        with col1:
            # UPDATED to use dashed-feedback-box and dash points
            st.markdown('<div class="dashed-feedback-box">', unsafe_allow_html=True)
            st.markdown("<h4>Good Points from Your CV</h4>", unsafe_allow_html=True)
            st.markdown('<ul class="feedback-list">', unsafe_allow_html=True)
            for point in strengths:
                st.markdown(f"<li>- {point}</li>", unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)
        with col2:
            # UPDATED to use dashed-feedback-box and dash points
            st.markdown('<div class="dashed-feedback-box">', unsafe_allow_html=True)
            st.markdown("<h4>Improvement Suggestions</h4>", unsafe_allow_html=True)
            st.markdown('<ul class="feedback-list">', unsafe_allow_html=True)
            for point in improvements:
                st.markdown(f"<li>- {point}</li>", unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)

    else:
        st.info("📤 Upload your resume and click 'Analyze My Resume' in the sidebar to see your dashboard.")

elif selected == "📄 Raw Text Viewer":
    st.subheader("📄 Raw Text Extracted from Your Resume")
    st.markdown("The AI uses this text for its analysis. If this looks incorrect, it might be due to complex formatting in your original PDF.")
    if st.session_state.resume_text:
        st.text_area("Resume Content:", st.session_state.resume_text, height=600)
    else:
        st.info("Upload a resume in the sidebar to view its raw text content.")

elif selected == "✨ AI Editor":
    st.subheader("✨ One-Click Resume Enhancement")
    st.markdown("Let our AI rewrite your resume for maximum impact. Review the changes side-by-side and download the improved version as a professional PDF.")
    st.markdown("---")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume in the sidebar first to use the AI Editor.")
    else:
        if st.button("🚀 Generate Enhanced Resume", use_container_width=True, type="primary"):
            with st.spinner("AI is rewriting your entire resume... This may take a minute."):
                prompt = ENHANCED_RESUME_PROMPT_TEMPLATE.format(resume_text=st.session_state.resume_text)
                enhanced_text = get_gemini_response(prompt, return_raw_text=True)
                if enhanced_text and len(enhanced_text) > 50:
                    st.session_state.enhanced_resume_text = enhanced_text
                else:
                    st.error("Could not generate the enhanced resume. The AI response was empty or invalid. Please try again.")
                    st.session_state.enhanced_resume_text = None

        if st.session_state.enhanced_resume_text:
            st.success("Your enhanced resume is ready! Compare the versions below.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Original Resume:", value=st.session_state.resume_text, height=500)
            with col2:
                st.text_area("AI-Enhanced Resume:", value=st.session_state.enhanced_resume_text, height=500)
            
            pdf_bytes = create_resume_pdf(st.session_state.enhanced_resume_text)
            
            st.download_button(
                label="⬇️ Download Enhanced Resume as PDF",
                data=pdf_bytes,
                file_name=f"Enhanced_{st.session_state.get('file_name', 'resume.pdf')}",
                mime="application/pdf",
                use_container_width=True
            )
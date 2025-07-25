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
    page_icon="📄",
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
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        st.error(f"An error occurred while communicating with the AI model: {e}")
        return None

def get_pdf_text(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        st.session_state.pdf_file_bytes = file_bytes
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def create_resume_pdf(text_content):
    """Generates a PDF from a string of text using built-in fonts."""
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
ANALYSIS_PROMPT_TEMPLATE = """
You are an expert ATS evaluator and resume coach. Review the resume with a professional recruiter's eye — even if it's well-written, your task is to **critically analyze it** and identify both strengths and areas for improvement.

Give a realistic score from based on the Analysis up to threshold value upto 93 and not more than that. Even for well-crafted resumes, point out subtle gaps in ATS optimization, formatting inconsistencies, lack of measurable impact, or missed keywords.

Resume Text: --- {resume_text} ---
Job Description: --- {job_description} ---

Respond in the following valid JSON format:
{{
  "overall_score": 78,
  "score_summary": "Good but Needs Optimization",
  "optimization_summary": "This resume is fairly strong, but there are several areas where even a good resume can fall short — especially in matching specific job descriptions and optimizing for ATS.",
  "whats_working_well": [
    "Well-structured format",
    "Relevant skills and tools included",
    "Professional tone maintained"
  ],
  "improvement_recommendations": [
    {{
      "title": "Not Tailored to Job Description",
      "suggestion": "Resume content appears generic. Mirror the specific language and keywords of the target job description to increase ATS compatibility.",
      "priority": "HIGH"
    }},
    {{
      "title": "Add More Measurable Achievements",
      "suggestion": "Even though the experience section is clear, you can make it more impactful by quantifying outcomes and results.",
      "priority": "HIGH"
    }},
    {{
      "title": "Refine Section Headers & Visual Hierarchy",
      "suggestion": "The formatting could use stronger headers or visual distinction between sections to improve readability and recruiter navigation.",
      "priority": "MEDIUM"
    }},
    {{
      "title": "Enhance Summary Section",
      "suggestion": "The summary can be more dynamic. Highlight 2–3 major strengths or career goals in a concise, attention-grabbing way.",
      "priority": "MEDIUM"
    }}
  ]
}}
"""

SECTIONAL_ANALYSIS_PROMPT_TEMPLATE = """
You are an expert resume parser and analyst. Your task is to break down the provided resume text into its constituent sections and provide structural metrics.
For the resume as a whole, provide:
1. The total word count.
2. The total number of lines.
For each section you identify (e.g., Contact Information, Summary, Experience, etc.), you must:
1. Extract the exact text content of that section.
2. Provide a quality score for that section from 0 to 100.
3. Provide a brief justification for the score.
The final output must be a single, valid JSON object and nothing else.
Resume Text: --- {resume_text} ---
Return your analysis in this exact JSON format:
{{
  "structure_analysis": {{
    "word_count": 587,
    "line_count": 45
  }},
  "sections": [
    {{ "section_name": "Contact Information", "score": 100, "content": "John Doe | New York, NY | 123-456-7890 | john.doe@email.com", "justification": "All essential contact details are present and clear." }},
    {{ "section_name": "Summary", "score": 90, "content": "A results-driven software engineer...", "justification": "Strong summary, but could be slightly more concise." }},
    {{ "section_name": "Experience", "score": 85, "content": "Software Engineer at TechCorp...", "justification": "Good description of roles, but bullet points could be strengthened with more quantifiable achievements." }},
    {{ "section_name": "Education", "score": 95, "content": "B.S. in Computer Science - University of Example", "justification": "Education is clearly listed and relevant."}},
    {{ "section_name": "Skills", "score": 92, "content": "Python, Java, SQL, React, AWS", "justification": "Good list of relevant technical skills."}}
  ]
}}"""

RESUME_DOMAIN_PROMPT = """
You are an expert resume classifier. Your task is to analyze the given resume text and identify:

1. The most likely professional or academic field it belongs to (e.g., IT, BBA, Engineering, Nursing, Design).
2. The education background (e.g., Bachelor's in IT, MBA, Diploma in Civil Engineering).
3. Notable tools, technologies, or keywords present (e.g., Python, Canva, Accounting Software).

Return your answer in a single valid JSON object like:
{{
  "predicted_field": "IT",
  "education_background": "Bachelor's in Computer Science",
  "notable_keywords_or_tools": ["Python", "React", "Git"]
}}

Resume:
---
{resume_text}
---
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
    .feature-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); text-align: center; border: 1px solid #E0E0E0; height: 100%; }
    .feature-card.highlight { background-color: #E8F5E9; border: 1px solid #A5D6A7; }
    .feature-card-icon { font-size: 2rem; margin-bottom: 10px; }
    .score-circle { position: relative; width: 200px; height: 200px; border-radius: 50%; background: conic-gradient(#4CAF50 var(--score-deg), #E0E0E0 0deg); display: flex; align-items: center; justify-content: center; margin: 20px auto; }
    .score-inner-circle { width: 170px; height: 170px; background: white; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .score-text { font-size: 1.2rem; font-weight: 600; color: #555; }
    .score-value { font-size: 3.5rem; font-weight: bold; color: #333; }
    .score-badge { background-color: #43A047; color: white; padding: 5px 15px; border-radius: 20px; font-weight: 600; margin-top: 5px; }
    .rec-box, .working-well-box { background-color: #F8F9FA; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #E0E0E0; }
    .rec-box.high { border-left-color: #F44336; } .rec-box.medium { border-left-color: #FFC107; } .rec-box.low { border-left-color: #4CAF50; }
    .rec-title { font-weight: bold; margin-bottom: 5px; }
    .priority-badge { float: right; font-size: 0.8rem; padding: 2px 8px; border-radius: 10px; color: white; }
    .priority-badge.HIGH { background-color: #F44336; } .priority-badge.MEDIUM { background-color: #FFC107; } .priority-badge.LOW { background-color: #4CAF50; }
    .section-card { background-color: white; border: 1px solid #e0e6ed; border-left: 4px solid #4361ee; border-radius: 8px; padding: 20px; margin-bottom: 15px; position: relative; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }
    .section-title { font-size: 1.2rem; font-weight: 600; color: #343a40; }
    .section-score-badge { position: absolute; top: 15px; right: 15px; background-color: #e8f5e9; color: #388e3c; padding: 5px 12px; border-radius: 15px; font-size: 0.9em; font-weight: 600; }
    .section-content { color: #495057; white-space: pre-wrap; font-family: 'monospace'; font-size: 0.9rem; }
    
    /* Styles for the structure analysis metrics */
    .st-emotion-cache-1fjr796 { gap: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetricValue"] { color: #0d6efd; } /* Sections Found */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetricValue"] { color: #198754; } /* Avg Score */
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetricValue"] { color: #007bff; } /* Word Count */
    div[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetricValue"] { color: #6f42c1; } /* Lines */
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "analysis_data" not in st.session_state: st.session_state.analysis_data = None
if "sectional_data" not in st.session_state: st.session_state.sectional_data = None
if "viewer_sub_tab" not in st.session_state: st.session_state.viewer_sub_tab = "Structured View"
if "enhanced_resume_text" not in st.session_state: st.session_state.enhanced_resume_text = None
if "resume_domain_data" not in st.session_state: st.session_state.resume_domain_data = None
if "resume_text" not in st.session_state: st.session_state.resume_text = None
if "file_name" not in st.session_state: st.session_state.file_name = None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50.png?text=Recruit+Nepal", width=150)
    st.title("CV Input")
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Target Job Description (Optional)", height=150)
    analyze_button = st.button("Analyze My Resume", use_container_width=True, type="primary")
    if uploaded_file: 
        st.session_state.resume_text = get_pdf_text(uploaded_file)
        st.session_state.file_name = uploaded_file.name

# --- Trigger Analysis ---
if analyze_button and 'resume_text' in st.session_state:
    st.session_state.enhanced_resume_text = None
    with st.spinner("🤖 Performing overall analysis..."):
        prompt1 = ANALYSIS_PROMPT_TEMPLATE.format(resume_text=st.session_state.resume_text, job_description=job_description or "N/A")
        response1 = get_gemini_response(prompt1)
        if response1:
            try: 
                st.session_state.analysis_data = json.loads(response1)
            except json.JSONDecodeError: 
                st.error("Failed to parse overall analysis.")
    
    with st.spinner("🤖 Breaking down resume sections..."):
        prompt2 = SECTIONAL_ANALYSIS_PROMPT_TEMPLATE.format(resume_text=st.session_state.resume_text)
        response2 = get_gemini_response(prompt2)
        if response2:
            try: 
                st.session_state.sectional_data = json.loads(response2)
            except json.JSONDecodeError: 
                st.error("Failed to parse sectional analysis.")
    
    with st.spinner("🎯 Predicting resume field and tools..."):
        if st.session_state.get("resume_text"):
            domain_prompt = RESUME_DOMAIN_PROMPT.format(resume_text=st.session_state.resume_text)
            domain_response = get_gemini_response(domain_prompt)
            if domain_response:
                try:
                    st.session_state.resume_domain_data = json.loads(domain_response)
                except json.JSONDecodeError:
                    st.warning("Could not parse domain prediction response.")

# --- Main Page UI ---
st.title("Recruit Nepal CV Analyzer")
st.write("Get an instant ATS-friendly score with personalized tips to improve your CV")
cols = st.columns([1, 1.1, 1])
with cols[0]: st.markdown('<div class="feature-card"><div class="feature-card-icon">🎯</div><div>Analyze compatibility with recruitment systems</div></div>', unsafe_allow_html=True)
with cols[1]: st.markdown('<div class="feature-card highlight"><div class="feature-card-icon">⚡</div><div>Get detailed suggestions immediately</div></div>', unsafe_allow_html=True)
with cols[2]: st.markdown('<div class="feature-card"><div class="feature-card-icon">🇳🇵</div><div>Tailored for Nepalese job market</div></div>', unsafe_allow_html=True)
st.write("")

# --- Tabbed Navigation ---
selected = option_menu(
    menu_title=None, 
    options=["Score & Analysis", "Resume Viewer", "AI Editor"],
    icons=["clipboard-data", "file-earmark-text", "magic"], 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"}, 
        "icon": {"color": "#6c757d", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}, 
        "nav-link-selected": {"background-color": "#4CAF50", "color": "white"}
    }
)

# --- Tab Content ---
if selected == "Score & Analysis":
    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        score = data.get("overall_score", 0)
        score_deg = score * 3.6
        st.markdown(
            f"""<div class='score-circle' style='--score-deg: {score_deg}deg;'>
                <div class='score-inner-circle'>
                    <div class='score-text'>Your ATS Score</div>
                    <div class='score-value'>{score}/100</div>
                    <div class='score-badge'>{data.get("score_summary", "Good")}</div>
                </div>
            </div>""", 
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='text-align: center;'>{data.get('optimization_summary', '')}</p>", unsafe_allow_html=True)
        st.progress(score / 100)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ What's Working Well")
            for item in data.get("whats_working_well", []): 
                st.markdown(f"<div class='working-well-box'>✔️ {item}</div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("💡 Improvement Recommendations")
            for rec in data.get("improvement_recommendations", []):
                p = rec.get('priority', 'LOW')
                st.markdown(
                    f"""<div class='rec-box {p.lower()}'>
                        <span class='priority-badge {p}'>{p}</span>
                        <div class='rec-title'>{rec.get('title')}</div>
                        <div>{rec.get('suggestion')}</div>
                    </div>""", 
                    unsafe_allow_html=True
                )
    else: 
        st.info("Upload your resume and click 'Analyze My Resume' in the sidebar to see your score.")

elif selected == "Resume Viewer":
    st.subheader(f"Resume Viewer - {st.session_state.get('file_name', 'No file uploaded')}")
    c1, c2, c3 = st.columns(3)
    if c1.button("Structured View"): st.session_state.viewer_sub_tab = "Structured View"
    if c2.button("Raw Text"): st.session_state.viewer_sub_tab = "Raw Text"
    if c3.button("Section Analysis"):
        st.session_state.viewer_sub_tab = "Section Analysis"
    st.markdown("---")

    if st.session_state.viewer_sub_tab == "Structured View":
        if st.session_state.sectional_data:
            # 🧠 AI Predicted Domain Display
            domain_data = st.session_state.get("resume_domain_data", {})
            if domain_data:
                st.markdown("### 🧠 AI-Predicted Resume Field & Tools")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"- **Predicted Field:** {domain_data.get('predicted_field', 'N/A')}")
                    st.markdown(f"- **Education Background:** {domain_data.get('education_background', 'N/A')}")
                with col2:
                    tools = ', '.join(domain_data.get("notable_keywords_or_tools", [])) or "N/A"
                    st.markdown(f"- **Tools / Keywords:** {tools}")
                st.markdown("---")

            # 📑 Resume Sections
            icon_map = {
                "Contact Information": "☎️", "Summary": "📝", "Experience": "💼",
                "Education": "🎓", "Skills": "🛠️", "Projects": "🚀"
            }
            
            for section in st.session_state.sectional_data.get("sections", []):
                name = section.get("section_name", "Unknown Section")
                st.markdown(
                    f"""
                    <div class='section-card'>
                        <div class='section-score-badge'>{section.get('score', 'N/A')}/100</div>
                        <div class='section-header'>
                            <span>{icon_map.get(name, '📄')}</span>
                            <span class='section-title'>{name}</span>
                        </div>
                        <div class='section-content'>{section.get('content', 'No content found.')}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("Run the analysis to see the structured view.")

    elif st.session_state.viewer_sub_tab == "Raw Text":
        if 'resume_text' in st.session_state: 
            st.text_area("Extracted Raw Text", st.session_state.resume_text, height=500)
        else: 
            st.info("Upload a resume to see the raw text.")

    elif st.session_state.viewer_sub_tab == "Section Analysis":
        if st.session_state.sectional_data:
            data = st.session_state.sectional_data
            st.subheader("Resume Structure Analysis")
            structure_stats = data.get('structure_analysis', {})
            sections = data.get('sections', [])
            sections_found = len(sections)
            all_scores = [s.get('score', 0) for s in sections if isinstance(s.get('score'), (int, float))]
            avg_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0
            word_count = structure_stats.get('word_count', len(st.session_state.resume_text.split()))
            line_count = structure_stats.get('line_count', len(st.session_state.resume_text.split('\n')))

            cols = st.columns(4)
            cols[0].metric("Sections Found", f"{sections_found}")
            cols[1].metric("Avg Score", f"{avg_score}")
            cols[2].metric("Word Count", f"{word_count}")
            cols[3].metric("Lines", f"{line_count}")
            st.markdown("---")
            
            st.subheader("Section Completeness:")
            for section in sections:
                name = section.get('section_name', 'Unknown')
                score = section.get('score', 0)
                justification = section.get('justification', 'No justification provided')
                
                st.markdown(f"### {name} ({score}/100)")
                st.markdown(f"**Justification:** {justification}")
                st.markdown(f"```\n{section.get('content', 'No content')}\n```")
                st.markdown("---")
        else: 
            st.info("Run the analysis to see the section-by-section analysis.")

elif selected == "AI Editor":
    st.subheader("✨ One-Click Resume Enhancement")
    st.write("Let our AI automatically rewrite your entire resume for maximum impact. Review the enhanced version and download it as a professional PDF.")
    st.markdown("---")
    
    if 'resume_text' not in st.session_state or not st.session_state.resume_text:
        st.warning("Please upload a resume in the sidebar first.")
    else:
        if st.button("🚀 Generate Enhanced Resume", use_container_width=True, type="primary"):
            with st.spinner("AI is rewriting your entire resume... This may take a minute."):
                prompt = ENHANCED_RESUME_PROMPT_TEMPLATE.format(resume_text=st.session_state.resume_text)
                enhanced_text = get_gemini_response(prompt)
                if enhanced_text and len(enhanced_text) > 50:
                    st.session_state.enhanced_resume_text = enhanced_text
                else:
                    st.error("Could not generate the enhanced resume. The AI response was empty or invalid. Please try again.")
                    st.session_state.enhanced_resume_text = None

        if st.session_state.enhanced_resume_text:
            st.success("Your enhanced resume is ready!")
            st.text_area("Review the Enhanced Text:", value=st.session_state.enhanced_resume_text, height=400)
            
            pdf_bytes = create_resume_pdf(st.session_state.enhanced_resume_text)
            
            st.download_button(
                label="⬇️ Download Enhanced Resume as PDF",
                data=pdf_bytes,
                file_name=f"Enhanced_{st.session_state.get('file_name', 'resume.pdf')}",
                mime="application/pdf",
                use_container_width=True
            )
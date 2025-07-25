import streamlit as st
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
import fitz  # PyMuPDF
from streamlit_pdf_viewer import pdf_viewer
from streamlit_option_menu import option_menu
from fpdf import FPDF
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import language_tool_python
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(
    page_title="Recruit Nepal CV Analyzer Pro",
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

# Initialize language tool for grammar checking
@st.cache_resource
def load_grammar_tool():
    try:
        return language_tool_python.LanguageTool('en-US')
    except:
        return None

grammar_tool = load_grammar_tool()

# Load spaCy model for NLP tasks
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("spaCy model not found. Some features may be limited.")
        return None

nlp = load_spacy_model()

# Industry keywords database
INDUSTRY_KEYWORDS = {
    "Technology": ["python", "java", "javascript", "react", "nodejs", "sql", "aws", "docker", "kubernetes", "api", "frontend", "backend", "fullstack", "devops", "agile", "scrum", "git", "machine learning", "ai", "data science", "cloud", "microservices"],
    "Healthcare": ["medical", "healthcare", "nursing", "patient care", "clinical", "diagnosis", "treatment", "pharmaceutical", "hospital", "medical records", "hipaa", "healthcare management", "medical device", "telemedicine"],
    "Finance": ["financial", "banking", "investment", "accounting", "finance", "risk management", "portfolio", "trading", "compliance", "audit", "tax", "budgeting", "financial analysis", "wealth management"],
    "Marketing": ["marketing", "digital marketing", "seo", "sem", "social media", "content marketing", "brand management", "campaign", "analytics", "conversion", "lead generation", "email marketing", "ppc"],
    "Education": ["teaching", "education", "curriculum", "instruction", "assessment", "classroom management", "educational technology", "student engagement", "pedagogy", "learning outcomes"],
    "Engineering": ["engineering", "mechanical", "electrical", "civil", "software", "design", "manufacturing", "quality assurance", "testing", "troubleshooting", "technical documentation", "project management"],
    "Sales": ["sales", "business development", "client relations", "revenue", "targets", "crm", "lead generation", "negotiation", "closing", "customer acquisition", "account management"],
    "Human Resources": ["hr", "human resources", "recruitment", "talent acquisition", "employee relations", "performance management", "training", "onboarding", "compensation", "benefits", "policy development"]
}

SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem-solving", "analytical", "creative", "adaptable", "organized", "detail-oriented", "time management", "interpersonal", "collaborative", "innovative", "strategic", "customer-focused", "results-oriented", "self-motivated", "multitasking", "critical thinking", "decision-making"]

BUZZWORDS = ["synergy", "leverage", "utilize", "facilitate", "optimize", "streamline", "enhance", "implement", "execute", "deliver", "drive", "manage", "oversee", "coordinate", "spearhead", "pioneer", "champion", "revolutionize", "transform", "innovate"]

# --- Enhanced Analysis Functions ---
def analyze_resume_structure(text):
    """Comprehensive resume structure analysis"""
    lines = text.split('\n')
    words = text.split()
    
    # Calculate readability scores
    readability_score = flesch_reading_ease(text)
    grade_level = flesch_kincaid_grade(text)
    
    # Detect formatting elements
    bullet_count = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '◦')))
    
    # Estimate pages (assuming 250 words per page)
    estimated_pages = max(1, len(words) // 250)
    
    # Check for tables/columns (simple heuristic)
    table_indicators = text.count('|') + text.count('\t')
    
    # Font consistency (check for mixed case patterns)
    uppercase_lines = sum(1 for line in lines if line.isupper() and len(line.strip()) > 1)
    
    return {
        "word_count": len(words),
        "line_count": len(lines),
        "resume_length_pages": estimated_pages,
        "readability_score": readability_score,
        "grade_level": grade_level,
        "bullet_usage": bullet_count,
        "table_indicators": table_indicators,
        "uppercase_headings": uppercase_lines
    }

def detect_grammar_spelling_errors(text):
    """Detect grammar and spelling errors"""
    if not grammar_tool:
        return {"grammar_errors": 0, "spelling_errors": 0, "error_details": []}
    
    try:
        matches = grammar_tool.check(text)
        grammar_errors = []
        spelling_errors = []
        
        for match in matches:
            if 'TYPOS' in match.ruleId or 'MORFOLOGIK' in match.ruleId:
                spelling_errors.append({
                    "error": match.message,
                    "context": match.context,
                    "suggestions": match.replacements[:3]
                })
            else:
                grammar_errors.append({
                    "error": match.message,
                    "context": match.context,
                    "suggestions": match.replacements[:3]
                })
        
        return {
            "grammar_errors": len(grammar_errors),
            "spelling_errors": len(spelling_errors),
            "grammar_details": grammar_errors[:10],  # Limit for display
            "spelling_details": spelling_errors[:10]
        }
    except Exception as e:
        return {"grammar_errors": 0, "spelling_errors": 0, "error_details": []}

def analyze_content_depth(text):
    """Analyze content depth and demonstration"""
    words = text.lower().split()
    
    # Count quantified achievements (numbers followed by relevant terms)
    quantified_patterns = [
        r'\d+%', r'\d+\s*percent', r'\$\d+', r'\d+\s*million', r'\d+\s*thousand',
        r'\d+\s*years?', r'\d+\s*months?', r'\d+\s*projects?', r'\d+\s*clients?',
        r'\d+\s*teams?', r'\d+\s*people', r'\d+\s*members', r'increased.*\d+',
        r'reduced.*\d+', r'improved.*\d+', r'managed.*\d+', r'led.*\d+'
    ]
    
    quantified_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in quantified_patterns)
    
    # Detect soft skills
    soft_skills_found = [skill for skill in SOFT_SKILLS if skill in text.lower()]
    
    # Detect technical skills (simple keyword matching)
    tech_keywords = []
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        tech_keywords.extend(keywords)
    
    tech_skills_found = [skill for skill in tech_keywords if skill in text.lower()]
    
    # Detect passive voice
    passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are', 'am']
    passive_count = sum(text.lower().count(indicator) for indicator in passive_indicators)
    
    # Check tense consistency
    past_tense_indicators = ['ed ', 'developed', 'managed', 'created', 'implemented', 'designed']
    present_tense_indicators = ['ing ', 'manage', 'create', 'implement', 'design', 'develop']
    
    past_tense_count = sum(text.lower().count(indicator) for indicator in past_tense_indicators)
    present_tense_count = sum(text.lower().count(indicator) for indicator in present_tense_indicators)
    
    return {
        "quantified_achievements": quantified_count,
        "soft_skills_found": soft_skills_found,
        "tech_skills_found": tech_skills_found,
        "soft_skills_count": len(soft_skills_found),
        "tech_skills_count": len(tech_skills_found),
        "passive_voice_count": passive_count,
        "past_tense_count": past_tense_count,
        "present_tense_count": present_tense_count,
        "tense_consistency_score": abs(past_tense_count - present_tense_count)
    }

def detect_industry_and_relevance(text, job_description=""):
    """Detect industry and calculate relevance scores"""
    text_lower = text.lower()
    job_lower = job_description.lower() if job_description else ""
    
    # Calculate industry scores
    industry_scores = {}
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        industry_scores[industry] = score
    
    # Determine primary industry
    detected_industry = max(industry_scores, key=industry_scores.get) if industry_scores else "General"
    
    # Calculate keyword density
    words = text_lower.split()
    total_keywords = sum(industry_scores.values())
    keyword_density = (total_keywords / len(words)) * 100 if words else 0
    
    # Job description matching
    job_match_score = 0
    if job_description:
        job_words = set(job_lower.split())
        resume_words = set(text_lower.split())
        common_words = job_words.intersection(resume_words)
        job_match_score = (len(common_words) / len(job_words)) * 100 if job_words else 0
    
    # Buzzword count
    buzzword_count = sum(1 for buzzword in BUZZWORDS if buzzword in text_lower)
    
    return {
        "detected_industry": detected_industry,
        "industry_scores": industry_scores,
        "keyword_density": keyword_density,
        "job_match_score": job_match_score,
        "buzzword_count": buzzword_count,
        "total_industry_keywords": total_keywords
    }

def calculate_ats_score(text):
    """Calculate ATS-friendly formatting score"""
    score = 100
    
    # Penalty for special characters that ATS might not handle
    special_chars = ['@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}']
    special_char_count = sum(text.count(char) for char in special_chars)
    score -= min(special_char_count * 2, 20)
    
    # Penalty for excessive formatting
    if text.count('\t') > 10:
        score -= 10
    
    # Bonus for standard section headers
    standard_headers = ['experience', 'education', 'skills', 'summary', 'objective', 'contact']
    header_bonus = sum(5 for header in standard_headers if header in text.lower())
    score += min(header_bonus, 30)
    
    return max(0, min(100, score))

def comprehensive_resume_analysis(text, job_description=""):
    """Perform comprehensive resume analysis"""
    
    # Structure analysis
    structure = analyze_resume_structure(text)
    
    # Grammar and spelling
    errors = detect_grammar_spelling_errors(text)
    
    # Content depth
    content = analyze_content_depth(text)
    
    # Industry relevance
    industry = detect_industry_and_relevance(text, job_description)
    
    # ATS score
    ats_score = calculate_ats_score(text)
    
    # Calculate composite scores
    grammar_score = max(0, 100 - (errors['grammar_errors'] * 5) - (errors['spelling_errors'] * 3))
    formatting_score = min(100, 70 + (structure['bullet_usage'] * 2) + (structure['uppercase_headings'] * 3))
    content_score = min(100, 50 + (content['quantified_achievements'] * 5) + (content['soft_skills_count'] * 2) + (content['tech_skills_count'] * 2))
    
    # Overall score calculation
    overall_score = (ats_score * 0.3 + grammar_score * 0.2 + formatting_score * 0.2 + 
                    content_score * 0.2 + industry['job_match_score'] * 0.1)
    
    return {
        "structure": structure,
        "errors": errors,
        "content": content,
        "industry": industry,
        "scores": {
            "overall_score": int(overall_score),
            "ats_score": ats_score,
            "grammar_score": grammar_score,
            "formatting_score": formatting_score,
            "content_score": content_score,
            "industry_match_score": industry['job_match_score']
        }
    }

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

    return pdf.output(dest='S')

def create_wordcloud(text):
    """Generate word cloud from resume text"""
    try:
        # Remove common stop words
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
        
        # Convert to base64 for display
        img_buffer = BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

# --- Enhanced Prompt Definitions ---
ANALYSIS_PROMPT_TEMPLATE = """
You are an expert ATS and a professional career coach from Nepal. Analyze the provided resume text.
Your response MUST be a single, valid JSON object. Provide a score from 0-100. Be critical but fair.

Resume Text: --- {resume_text} ---
Job Description: --- {job_description} ---

Consider these factors in your analysis:
1. ATS compatibility and formatting
2. Industry relevance and keyword optimization
3. Content quality and quantified achievements
4. Professional presentation and clarity
5. Completeness of essential sections

Return your analysis in this exact JSON format:
{{
  "overall_score": 85,
  "score_summary": "Excellent",
  "optimization_summary": "Your resume is well-structured and optimized for most ATS systems.",
  "whats_working_well": ["Clear contact information", "Strong summary section", "Good use of keywords for the industry", "Quantified achievements present", "Professional formatting"],
  "improvement_recommendations": [
    {{"title": "Quantify Achievements", "suggestion": "Add more specific metrics to demonstrate impact. For example, 'Increased sales by 20%' instead of 'Increased sales'.", "priority": "HIGH"}},
    {{"title": "Action Verbs", "suggestion": "Start each bullet point with strong action verbs like 'Orchestrated', 'Engineered', or 'Maximized'.", "priority": "MEDIUM"}},
    {{"title": "Industry Keywords", "suggestion": "Include more industry-specific keywords to improve ATS matching.", "priority": "MEDIUM"}}
  ],
  "industry_insights": {{
    "detected_industry": "Technology",
    "relevance_score": 78,
    "missing_keywords": ["cloud computing", "microservices", "agile methodology"]
  }}
}}"""

SECTIONAL_ANALYSIS_PROMPT_TEMPLATE = """
You are an expert resume parser and analyst. Analyze the provided resume text and break it down into sections.

Resume Text: --- {resume_text} ---

Return your analysis in this exact JSON format:
{{
  "structure_analysis": {{
    "word_count": 587,
    "line_count": 45,
    "estimated_pages": 2,
    "readability_score": 65.2
  }},
  "sections": [
    {{ "section_name": "Contact Information", "score": 100, "content": "John Doe | New York, NY | 123-456-7890 | john.doe@email.com", "justification": "All essential contact details are present and clear." }},
    {{ "section_name": "Summary", "score": 90, "content": "A results-driven software engineer...", "justification": "Strong summary that highlights key qualifications and career focus." }},
    {{ "section_name": "Experience", "score": 85, "content": "Software Engineer at TechCorp...", "justification": "Good experience descriptions with some quantified achievements, but could be strengthened further." }},
    {{ "section_name": "Education", "score": 95, "content": "B.S. in Computer Science - University of Example", "justification": "Education is clearly presented with relevant degree information." }},
    {{ "section_name": "Skills", "score": 92, "content": "Python, Java, SQL, React, AWS", "justification": "Comprehensive list of relevant technical skills for the industry." }}
  ]
}}"""

# --- Enhanced UI Styling ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F5; }
    #MainMenu, footer, header {visibility: hidden;}
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .feature-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
        text-align: center; 
        border: 1px solid #E0E0E0; 
        height: 100%; 
    }
    
    .feature-card.highlight { 
        background-color: #E8F5E9; 
        border: 1px solid #A5D6A7; 
    }
    
    .score-circle { 
        position: relative; 
        width: 200px; 
        height: 200px; 
        border-radius: 50%; 
        background: conic-gradient(#4CAF50 var(--score-deg), #E0E0E0 0deg); 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        margin: 20px auto; 
    }
    
    .score-inner-circle { 
        width: 170px; 
        height: 170px; 
        background: white; 
        border-radius: 50%; 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
    }
    
    .analysis-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    
    .error-card {
        background: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-card {
        background: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "analysis_data" not in st.session_state: 
    st.session_state.analysis_data = None
if "sectional_data" not in st.session_state: 
    st.session_state.sectional_data = None
if "comprehensive_data" not in st.session_state: 
    st.session_state.comprehensive_data = None
if "enhanced_resume_text" not in st.session_state: 
    st.session_state.enhanced_resume_text = None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.image("https://i.imgur.com/uJz103Z.png", width=150)
    st.title("CV Input")
    
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Target Job Description (Optional)", height=150)
    
    analyze_button = st.button("🚀 Analyze My Resume", use_container_width=True, type="primary")
    
    if uploaded_file: 
        st.session_state.resume_text = get_pdf_text(uploaded_file)
        st.session_state.file_name = uploaded_file.name
        st.success(f"✅ Loaded: {uploaded_file.name}")

# --- Trigger Analysis ---
if analyze_button and 'resume_text' in st.session_state:
    st.session_state.enhanced_resume_text = None
    
    with st.spinner("🤖 Performing comprehensive analysis..."):
        # Comprehensive analysis
        comprehensive_analysis = comprehensive_resume_analysis(
            st.session_state.resume_text, 
            job_description
        )
        st.session_state.comprehensive_data = comprehensive_analysis
        
        # AI-powered analysis
        prompt1 = ANALYSIS_PROMPT_TEMPLATE.format(
            resume_text=st.session_state.resume_text, 
            job_description=job_description or "N/A"
        )
        response1 = get_gemini_response(prompt1)
        if response1:
            try: 
                st.session_state.analysis_data = json.loads(response1)
            except json.JSONDecodeError: 
                st.error("Failed to parse AI analysis.")
        
        # Sectional analysis
        prompt2 = SECTIONAL_ANALYSIS_PROMPT_TEMPLATE.format(
            resume_text=st.session_state.resume_text
        )
        response2 = get_gemini_response(prompt2)
        if response2:
            try: 
                st.session_state.sectional_data = json.loads(response2)
            except json.JSONDecodeError: 
                st.error("Failed to parse sectional analysis.")

# --- Main Page UI ---
st.title("🎯 Recruit Nepal CV Analyzer Pro")
st.markdown("**Advanced ATS-friendly analysis with comprehensive insights for the Nepalese job market**")

# Feature cards
cols = st.columns([1, 1, 1, 1])
with cols[0]: 
    st.markdown('<div class="feature-card"><div class="feature-card-icon">🎯</div><div>ATS Optimization</div></div>', unsafe_allow_html=True)
with cols[1]: 
    st.markdown('<div class="feature-card highlight"><div class="feature-card-icon">📊</div><div>Detailed Analytics</div></div>', unsafe_allow_html=True)
with cols[2]: 
    st.markdown('<div class="feature-card"><div class="feature-card-icon">🔍</div><div>Industry Matching</div></div>', unsafe_allow_html=True)
with cols[3]: 
    st.markdown('<div class="feature-card"><div class="feature-card-icon">✨</div><div>AI Enhancement</div></div>', unsafe_allow_html=True)

st.markdown("---")

# --- Enhanced Tabbed Navigation ---
selected = option_menu(
    menu_title=None, 
    options=["📊 Dashboard", "🎯 Score Analysis", "📄 Resume Viewer", "🔍 Deep Analysis", "✨ AI Editor"],
    icons=["speedometer2", "clipboard-data", "file-earmark-text", "search", "magic"], 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"}, 
        "icon": {"color": "#6c757d", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}, 
        "nav-link-selected": {"background-color": "#4CAF50", "color": "white"}
    }
)

# --- Tab Content ---
if selected == "📊 Dashboard":
    if st.session_state.comprehensive_data:
        data = st.session_state.comprehensive_data
        
        st.subheader("📊 Resume Performance Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data['scores']['overall_score']}</h3>
                <p>Overall Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data['scores']['ats_score']}</h3>
                <p>ATS Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data['structure']['word_count']}</h3>
                <p>Word Count</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data['industry']['detected_industry']}</h3>
                <p>Industry</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Score Breakdown")
            
            scores_df = pd.DataFrame({
                'Category': ['ATS', 'Grammar', 'Formatting', 'Content', 'Industry Match'],
                'Score': [
                    data['scores']['ats_score'],
                    data['scores']['grammar_score'],
                    data['scores']['formatting_score'],
                    data['scores']['content_score'],
                    data['scores']['industry_match_score']
                ]
            })
            
            fig = px.bar(scores_df, x='Category', y='Score', 
                        title='Score Breakdown by Category',
                        color='Score',
                        color_continuous_scale='Viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Industry Keywords")
            
            industry_scores = data['industry']['industry_scores']
            industry_df = pd.DataFrame({
                'Industry': list(industry_scores.keys()),
                'Keywords Found': list(industry_scores.values())
            })
            
            fig = px.pie(industry_df, values='Keywords Found', names='Industry',
                        title='Industry Keyword Distribution')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Word Cloud
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔤 Resume Word Cloud")
        wordcloud_img = create_wordcloud(st.session_state.resume_text)
        if wordcloud_img:
            st.markdown(f'<img src="data:image/png;base64,{wordcloud_img}" style="width:100%">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("📁 Upload your resume and click 'Analyze My Resume' to see the dashboard.")

elif selected == "🎯 Score Analysis":
    if st.session_state.analysis_data and st.session_state.comprehensive_data:
        ai_data = st.session_state.analysis_data
        comp_data = st.session_state.comprehensive_data
        
        # Main score display
        score = ai_data.get("overall_score", 0)
        score_deg = score * 3.6
        
        st.markdown(f"""
        <div class='score-circle' style='--score-deg: {score_deg}deg;'>
            <div class='score-inner-circle'>
                <div class='score-text'>Your ATS Score</div>
                <div class='score-value'>{score}/100</div>
                <div class='score-badge'>{ai_data.get("score_summary", "Good")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>{ai_data.get('optimization_summary', '')}</p>", unsafe_allow_html=True)
        
        # Detailed scores
        st.markdown("---")
        st.subheader("📊 Detailed Score Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.metric("Grammar Score", f"{comp_data['scores']['grammar_score']}/100")
            st.write(f"**Grammar Errors:** {comp_data['errors']['grammar_errors']}")
            st.write(f"**Spelling Errors:** {comp_data['errors']['spelling_errors']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.metric("Content Score", f"{comp_data['scores']['content_score']}/100")
            st.write(f"**Quantified Achievements:** {comp_data['content']['quantified_achievements']}")
            st.write(f"**Soft Skills Found:** {comp_data['content']['soft_skills_count']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.metric("ATS Compatibility", f"{comp_data['scores']['ats_score']}/100")
            st.write(f"**Industry Match:** {comp_data['scores']['industry_match_score']:.1f}%")
            st.write(f"**Keyword Density:** {comp_data['industry']['keyword_density']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # What's working well and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ What's Working Well")
            for item in ai_data.get("whats_working_well", []):
                st.markdown(f'<div class="success-card">✔️ {item}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("💡 Improvement Recommendations")
            for rec in ai_data.get("improvement_recommendations", []):
                priority = rec.get('priority', 'LOW')
                color = {'HIGH': '#F44336', 'MEDIUM': '#FFC107', 'LOW': '#4CAF50'}[priority]
                st.markdown(f"""
                <div class="error-card" style="border-left-color: {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{rec.get('title', 'Recommendation')}</strong>
                        <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;">{priority}</span>
                    </div>
                    <p style="margin: 8px 0 0 0;">{rec.get('suggestion', '')}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📁 Upload your resume and analyze it to see detailed scores.")

elif selected == "📄 Resume Viewer":
    st.subheader(f"📄 Resume Viewer - {st.session_state.get('file_name', 'No file uploaded')}")
    
    # Sub-navigation
    viewer_tabs = st.tabs(["📋 Structured View", "📝 Raw Text", "🔍 Section Analysis", "📊 Structure Metrics"])
    
    with viewer_tabs[0]:  # Structured View
        if st.session_state.sectional_data:
            sections = st.session_state.sectional_data.get("sections", [])
            icon_map = {
                "Contact Information": "☎️", "Summary": "📝", "Experience": "💼", 
                "Education": "🎓", "Skills": "🛠️", "Projects": "🚀",
                "Certifications": "🏆", "Achievements": "⭐"
            }
            
            for section in sections:
                name = section.get("section_name", "Unknown Section")
                score = section.get("score", 0)
                
                # Color coding based on score
                if score >= 90:
                    border_color = "#4CAF50"
                elif score >= 70:
                    border_color = "#FFC107"
                else:
                    border_color = "#F44336"
                
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e0e6ed; border-left: 4px solid {border_color}; 
                           border-radius: 8px; padding: 20px; margin: 15px 0; position: relative;">
                    <div style="position: absolute; top: 15px; right: 15px; background: #e8f5e9; 
                               color: #388e3c; padding: 5px 12px; border-radius: 15px; font-weight: 600;">
                        {score}/100
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                        <span style="font-size: 1.5rem;">{icon_map.get(name, '📄')}</span>
                        <h3 style="margin: 0; color: #343a40;">{name}</h3>
                    </div>
                    <div style="color: #495057; white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; margin-bottom: 10px;">
                        {section.get('content', 'No content found.')}
                    </div>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-style: italic; color: #6c757d;">
                        <strong>Analysis:</strong> {section.get('justification', 'No analysis available.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run the analysis to see the structured view.")
    
    with viewer_tabs[1]:  # Raw Text
        if 'resume_text' in st.session_state:
            st.text_area("📝 Extracted Raw Text", st.session_state.resume_text, height=500)
        else:
            st.info("Upload a resume to see the raw text.")
    
    with viewer_tabs[2]:  # Section Analysis
        if st.session_state.sectional_data:
            data = st.session_state.sectional_data
            structure_stats = data.get('structure_analysis', {})
            sections = data.get('sections', [])
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("📑 Sections Found", len(sections))
            
            all_scores = [s.get('score', 0) for s in sections if isinstance(s.get('score'), (int, float))]
            avg_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0
            col2.metric("📊 Average Score", f"{avg_score}/100")
            
            col3.metric("📝 Word Count", structure_stats.get('word_count', 0))
            col4.metric("📄 Lines", structure_stats.get('line_count', 0))
            
            # Section completeness
            st.markdown("---")
            st.subheader("📋 Section Completeness")
            
            required_sections = ["Contact Information", "Summary", "Experience", "Education", "Skills"]
            found_sections = [s.get('section_name', '') for s in sections]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Present Sections:**")
                for section in found_sections:
                    st.markdown(f"✔️ {section}")
            
            with col2:
                missing_sections = [s for s in required_sections if s not in found_sections]
                if missing_sections:
                    st.markdown("**❌ Missing Sections:**")
                    for section in missing_sections:
                        st.markdown(f"❌ {section}")
                else:
                    st.markdown("**🎉 All essential sections present!**")
                    
        else:
            st.info("Run the analysis to see section analysis.")
    
    with viewer_tabs[3]:  # Structure Metrics
        if st.session_state.comprehensive_data:
            data = st.session_state.comprehensive_data
            structure = data['structure']
            
            # Structure metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("📊 Document Structure")
                st.metric("Word Count", structure['word_count'])
                st.metric("Estimated Pages", structure['resume_length_pages'])
                st.metric("Readability Score", f"{structure['readability_score']:.1f}")
                st.metric("Bullet Points Used", structure['bullet_usage'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("📝 Content Analysis")
                st.metric("Line Count", structure['line_count'])
                st.metric("Uppercase Headings", structure['uppercase_headings'])
                st.metric("Table Indicators", structure['table_indicators'])
                st.metric("Grade Level", f"{structure['grade_level']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Reading level interpretation
            st.markdown("---")
            st.subheader("📖 Readability Analysis")
            
            readability = structure['readability_score']
            if readability >= 60:
                level_desc = "Easy to read - Good for most audiences"
                level_color = "#4CAF50"
            elif readability >= 30:
                level_desc = "Fairly difficult - Standard for professional documents"
                level_color = "#FFC107"
            else:
                level_desc = "Very difficult - May need simplification"
                level_color = "#F44336"
            
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid {level_color}; padding: 15px; border-radius: 8px;">
                <strong>Readability Level:</strong> {level_desc}<br>
                <strong>Score:</strong> {readability:.1f}/100<br>
                <strong>Grade Level:</strong> {structure['grade_level']:.1f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run the analysis to see structure metrics.")

elif selected == "🔍 Deep Analysis":
    if st.session_state.comprehensive_data:
        data = st.session_state.comprehensive_data
        
        st.subheader("🔍 Deep Content Analysis")
        
        # Content analysis tabs
        deep_tabs = st.tabs(["🎯 Keywords & Industry", "📊 Content Quality", "⚠️ Issues & Errors", "🧠 Skills Analysis"])
        
        with deep_tabs[0]:  # Keywords & Industry
            industry_data = data['industry']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("🏭 Industry Analysis")
                st.metric("Detected Industry", industry_data['detected_industry'])
                st.metric("Keyword Density", f"{industry_data['keyword_density']:.2f}%")
                st.metric("Job Match Score", f"{industry_data['job_match_score']:.1f}%")
                st.metric("Buzzwords Count", industry_data['buzzword_count'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("📊 Industry Keyword Breakdown")
                
                industry_scores = industry_data['industry_scores']
                for industry, score in sorted(industry_scores.items(), key=lambda x: x[1], reverse=True):
                    if score > 0:
                        percentage = (score / max(industry_scores.values())) * 100
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>{industry}</span>
                                <span>{score} keywords</span>
                            </div>
                            <div style="background: #e0e0e0; height: 8px; border-radius: 4px; margin-top: 5px;">
                                <div style="background: #4CAF50; height: 100%; width: {percentage}%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with deep_tabs[1]:  # Content Quality
            content_data = data['content']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("📈 Achievement Quantification")
                st.metric("Quantified Achievements", content_data['quantified_achievements'])
                st.metric("Passive Voice Usage", content_data['passive_voice_count'])
                st.metric("Tense Consistency Score", content_data['tense_consistency_score'])
                
                # Tense analysis
                st.markdown("**Tense Usage:**")
                st.write(f"Past tense: {content_data['past_tense_count']}")
                st.write(f"Present tense: {content_data['present_tense_count']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("🛠️ Skills Distribution")
                
                # Skills breakdown chart
                skills_data = {
                    'Technical Skills': content_data['tech_skills_count'],
                    'Soft Skills': content_data['soft_skills_count']
                }
                
                fig = px.pie(
                    values=list(skills_data.values()),
                    names=list(skills_data.keys()),
                    title="Skills Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with deep_tabs[2]:  # Issues & Errors
            errors_data = data['errors']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("⚠️ Writing Issues")
                st.metric("Grammar Errors", errors_data['grammar_errors'])
                st.metric("Spelling Errors", errors_data['spelling_errors'])
                
                # Show sample errors if available
                if 'grammar_details' in errors_data and errors_data['grammar_details']:
                    st.markdown("**Sample Grammar Issues:**")
                    for error in errors_data['grammar_details'][:3]:
                        st.markdown(f"• {error.get('error', 'Grammar issue detected')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("🔧 Improvement Areas")
                
                # Show sample spelling errors if available
                if 'spelling_details' in errors_data and errors_data['spelling_details']:
                    st.markdown("**Spelling Issues:**")
                    for error in errors_data['spelling_details'][:3]:
                        st.markdown(f"• {error.get('error', 'Spelling issue detected')}")
                        if error.get('suggestions'):
                            st.markdown(f"  *Suggestions: {', '.join(error['suggestions'][:2])}*")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with deep_tabs[3]:  # Skills Analysis
            content_data = data['content']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("🧠 Soft Skills Found")
                
                if content_data['soft_skills_found']:
                    for skill in content_data['soft_skills_found'][:10]:
                        st.markdown(f"✅ {skill.title()}")
                else:
                    st.markdown("❌ No soft skills explicitly mentioned")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.subheader("💻 Technical Skills Found")
                
                if content_data['tech_skills_found']:
                    for skill in content_data['tech_skills_found'][:10]:
                        st.markdown(f"✅ {skill.title()}")
                else:
                    st.markdown("❌ No technical skills explicitly mentioned")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("📁 Upload your resume and analyze it to see deep analysis.")

elif selected == "✨ AI Editor":
    st.subheader("✨ AI-Powered Resume Enhancement")
    st.markdown("Transform your resume with advanced AI optimization for maximum impact and ATS compatibility.")
    
    if 'resume_text' not in st.session_state or not st.session_state.resume_text:
        st.warning("📁 Please upload a resume in the sidebar first.")
    else:
        # Enhancement options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("🎯 Enhancement Options")
            
            enhance_grammar = st.checkbox("✅ Fix Grammar & Spelling", value=True)
            enhance_keywords = st.checkbox("🔍 Optimize Keywords", value=True)
            enhance_format = st.checkbox("📝 Improve Formatting", value=True)
            enhance_impact = st.checkbox("📈 Strengthen Impact Statements", value=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("📊 Current Resume Stats")
            
            if st.session_state.comprehensive_data:
                data = st.session_state.comprehensive_data
                st.metric("Current Score", f"{data['scores']['overall_score']}/100")
                st.metric("Word Count", data['structure']['word_count'])
                st.metric("Grammar Issues", data['errors']['grammar_errors'])
                st.metric("Quantified Achievements", data['content']['quantified_achievements'])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhancement button
        if st.button("🚀 Generate Enhanced Resume", use_container_width=True, type="primary"):
            enhancement_options = {
                "grammar": enhance_grammar,
                "keywords": enhance_keywords,
                "format": enhance_format,
                "impact": enhance_impact
            }
            
            with st.spinner("🤖 AI is enhancing your resume... This may take a moment."):
                # Enhanced prompt with options
                enhancement_prompt = f"""
                You are an elite resume optimization AI. Enhance the following resume with these specific improvements:
                
                {'- Fix all grammar and spelling errors' if enhance_grammar else ''}
                {'- Optimize industry keywords and ATS compatibility' if enhance_keywords else ''}
                {'- Improve formatting and structure' if enhance_format else ''}
                {'- Strengthen impact statements with quantified achievements' if enhance_impact else ''}
                
                Original Resume:
                ---
                {st.session_state.resume_text}
                ---
                
                Job Description Context:
                {job_description or "General optimization"}
                
                Return ONLY the enhanced resume text, preserving the original structure while implementing the requested improvements.
                """
                
                enhanced_text = get_gemini_response(enhancement_prompt)
                if enhanced_text and len(enhanced_text) > 50:
                    st.session_state.enhanced_resume_text = enhanced_text
                    st.success("✅ Your enhanced resume is ready!")
                else:
                    st.error("❌ Could not generate enhanced resume. Please try again.")
        
        # Display enhanced resume
        if st.session_state.enhanced_resume_text:
            st.markdown("---")
            st.subheader("📝 Enhanced Resume Preview")
            
            # Comparison tabs
            comparison_tabs = st.tabs(["✨ Enhanced Version", "📊 Before vs After", "⬇️ Download"])
            
            with comparison_tabs[0]:
                st.text_area("Enhanced Resume:", value=st.session_state.enhanced_resume_text, height=500)
            
            with comparison_tabs[1]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Resume:**")
                    st.text_area("Original:", value=st.session_state.resume_text[:1000] + "...", height=300, disabled=True)
                
                with col2:
                    st.markdown("**Enhanced Resume:**")
                    st.text_area("Enhanced:", value=st.session_state.enhanced_resume_text[:1000] + "...", height=300, disabled=True)
            
            with comparison_tabs[2]:
                st.markdown("### 📥 Download Options")
                
                # Generate PDF
                pdf_bytes = create_resume_pdf(st.session_state.enhanced_resume_text)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_bytes,
                        file_name=f"Enhanced_{st.session_state.get('file_name', 'resume')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📝 Download as Text",
                        data=st.session_state.enhanced_resume_text,
                        file_name=f"Enhanced_{st.session_state.get('file_name', 'resume')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Re-analyze option
                if st.button("🔄 Re-analyze Enhanced Resume", use_container_width=True):
                    st.session_state.resume_text = st.session_state.enhanced_resume_text
                    st.session_state.file_name = f"Enhanced_{st.session_state.get('file_name', 'resume')}"
                    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🇳🇵 <strong>Recruit Nepal CV Analyzer Pro</strong> - Empowering Nepalese professionals with AI-driven resume optimization</p>
    <p>Built with ❤️ for the Nepalese job market | Powered by Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)
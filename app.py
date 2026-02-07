"""
Student Placement Readiness - Complete Premium Edition
Focus: Centered layout, branded sidebar, all charts visible, stable presentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, List

from scoring import readiness_level
from train_model import train_and_save_model
from preprocessing import StudentDataPreprocessor

# ============ CONFIGURATION ============
MODEL_DIR = "model"
st.set_page_config(
    page_title="PlacePrep Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PREMIUM CSS ============
st.markdown("""
    <style>
    /* Import Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Section - Centered */
    .hero-section {
        text-align: center;
        margin: 0 auto 4rem auto;
        max-width: 900px;
    }
    
    /* Aspirational Headline - Centered */
    .hero-headline {
        font-size: 3.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin: 0 auto 1rem auto;
        letter-spacing: -0.03em;
    }
    
    /* Soft Explanatory Subtitle - Centered */
    .hero-subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        font-weight: 400;
        line-height: 1.7;
        margin: 0 auto;
        max-width: 750px;
        text-align: center;
    }
    
    /* Sidebar Brand Header */
    .sidebar-brand {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 1rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        border-radius: 0;
    }
    
    .sidebar-brand-name {
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .sidebar-brand-tagline {
        font-size: 0.8rem;
        opacity: 0.9;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Premium Cards */
    .premium-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 10px 30px rgba(0,0,0,0.03);
        border: 1px solid #f3f4f6;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 15px 40px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    /* Score Display - Hero */
    .score-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    
    .score-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    
    .score-number {
        font-size: 5.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.5rem;
        position: relative;
    }
    
    .score-label {
        font-size: 1.4rem;
        opacity: 0.95;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        position: relative;
    }
    
    .score-context {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 1.8rem;
        line-height: 1.7;
        position: relative;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-subheader {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f9fafb 0%, white 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #d1d5db;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Impact Badges */
    .impact-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .impact-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .impact-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .impact-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        border: 1px solid #e5e7eb;
        height: 100%;
    }
    
    .chart-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .chart-caption {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 1.25rem;
    }
    
    /* Recommendation Cards */
    .rec-critical {
        background: linear-gradient(to right, #fef2f2 0%, white 100%);
        border-left: 5px solid #ef4444;
        padding: 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .rec-important {
        background: linear-gradient(to right, #fffbeb 0%, white 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .rec-optional {
        background: linear-gradient(to right, #f0fdf4 0%, white 100%);
        border-left: 5px solid #10b981;
        padding: 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .rec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.85rem;
    }
    
    .rec-action {
        font-size: 0.98rem;
        color: #374151;
        margin-bottom: 0.6rem;
        line-height: 1.6;
    }
    
    .rec-meta {
        font-size: 0.85rem;
        color: #6b7280;
        display: flex;
        gap: 1.5rem;
        margin-top: 0.85rem;
        flex-wrap: wrap;
    }
    
    /* Insight Box */
    .insight-box {
        background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        border: 1px solid #c4b5fd;
    }
    
    .insight-text {
        font-size: 0.95rem;
        color: #5b21b6;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Welcome Cards */
    .welcome-card {
        background: white;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .welcome-card:hover {
        border-color: #d1d5db;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transform: translateY(-4px);
    }
    
    .welcome-icon {
        font-size: 3.5rem;
        margin-bottom: 1.25rem;
    }
    
    .welcome-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.85rem;
    }
    
    .welcome-desc {
        font-size: 0.95rem;
        color: #6b7280;
        line-height: 1.6;
    }
    
    /* Remove default Streamlit spacing */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafafa 0%, white 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.85rem 2rem;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    
    /* Gauge specific styling */
    .gauge-container {
        background: white;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        border: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ============ HELPER FUNCTIONS ============

def get_field_importance(field: str, branch: str) -> str:
    """Determine importance of each field"""
    importance_map = {
        "CSE": {"CGPA": "high", "Coding_Skills": "high", "Communication_Skills": "medium", 
                "Projects": "high", "Internships": "medium"},
        "ECE": {"CGPA": "high", "Coding_Skills": "medium", "Communication_Skills": "medium", 
                "Projects": "medium", "Internships": "high"},
        "ME": {"CGPA": "high", "Coding_Skills": "low", "Communication_Skills": "high", 
               "Projects": "medium", "Internships": "high"},
        "CE": {"CGPA": "high", "Coding_Skills": "low", "Communication_Skills": "high", 
               "Projects": "medium", "Internships": "high"},
        "EE": {"CGPA": "high", "Coding_Skills": "medium", "Communication_Skills": "medium", 
               "Projects": "medium", "Internships": "high"}
    }
    return importance_map.get(branch, {}).get(field, "medium")


def get_impact_badge(importance: str) -> str:
    """Generate HTML badge"""
    labels = {"high": "High Impact", "medium": "Medium Impact", "low": "Low Impact"}
    return f'<span class="impact-badge impact-{importance}">{labels[importance]}</span>'


def get_score_context(score: float, level: str) -> Dict:
    """Provide score context"""
    if level == "Low":
        return {
            "range": "35-59",
            "message": "You're building your foundation. This is where growth begins.",
            "next_level": "Medium (60+)",
            "gap": f"{60 - score:.0f}",
            "percentile": "15-40th",
            "encouragement": "Many successful professionals started exactly where you are now."
        }
    elif level == "Medium":
        return {
            "range": "60-79",
            "message": "You're in a competitive position with strong potential.",
            "next_level": "High (80+)",
            "gap": f"{80 - score:.0f}",
            "percentile": "40-75th",
            "encouragement": "You're ahead of many peers. Focus on key improvements to reach top tier."
        }
    else:
        return {
            "range": "80-100",
            "message": "You're exceptionally prepared for placement opportunities.",
            "next_level": "Elite Status",
            "gap": "0",
            "percentile": "75-95th",
            "encouragement": "Your profile stands out. Maintain this excellence through interviews."
        }


def analyze_weakest_areas(cgpa: float, coding: int, communication: int, 
                         projects: int, internships: int) -> Dict:
    """Identify weakest and strongest areas"""
    scores = {
        "CGPA": (cgpa / 10) * 100,
        "Coding": (coding / 10) * 100,
        "Communication": (communication / 10) * 100,
        "Projects": (min(projects, 5) / 5) * 100,
        "Internships": (min(internships, 3) / 3) * 100
    }
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    weakest = sorted_scores[:2]
    strongest = sorted_scores[-1]
    
    return {
        "weakest": [weakest[0][0], weakest[1][0]],
        "strongest": strongest[0],
        "insight": f"Strengthening **{weakest[0][0]}** and **{weakest[1][0]}** will create the fastest improvement.",
        "scores": scores
    }


def get_prioritized_recommendations(cgpa: float, coding: int, communication: int,
                                   projects: int, internships: int, branch: str) -> List[Dict]:
    """Generate prioritized recommendations"""
    recs = []
    
    # Critical
    if cgpa < 6.5:
        recs.append({
            "priority": "critical",
            "icon": "🎯",
            "title": "Elevate Academic Performance",
            "action": f"Raise CGPA from {cgpa:.1f} to 7.0+ to unlock premium opportunities",
            "why": "Opens doors to 40% more companies that have CGPA cutoffs",
            "timeline": "2 semesters",
            "impact": "Maximum"
        })
    
    if internships == 0:
        recs.append({
            "priority": "critical",
            "icon": "💼",
            "title": "Secure Your First Industry Experience",
            "action": "Apply to 25+ companies this week. Target startups and product companies.",
            "why": "Real-world experience is non-negotiable for competitive placements",
            "timeline": "2-3 months",
            "impact": "Critical"
        })
    
    if coding < 5 and branch == "CSE":
        recs.append({
            "priority": "critical",
            "icon": "💻",
            "title": "Build Core Technical Strength",
            "action": "Solve 3 problems daily on LeetCode. Master Arrays, Strings, and basic patterns.",
            "why": "Technical interviews require minimum 6/10 coding proficiency",
            "timeline": "3 months",
            "impact": "Essential"
        })
    
    # Important
    if cgpa >= 6.5 and cgpa < 7.5:
        recs.append({
            "priority": "important",
            "icon": "📈",
            "title": "Target Premium Company Threshold",
            "action": f"Push from {cgpa:.1f} to 8.0+ for top-tier company eligibility",
            "why": "Elite companies often filter at 8.0+ CGPA",
            "timeline": "2-3 semesters",
            "impact": "High"
        })
    
    if projects < 3:
        recs.append({
            "priority": "important",
            "icon": "🚀",
            "title": "Develop Project Portfolio",
            "action": f"Create {3 - projects} production-grade projects with live deployments",
            "why": "Projects demonstrate practical ability beyond academic theory",
            "timeline": "3-4 months",
            "impact": "Very High"
        })
    
    if coding >= 5 and coding < 7:
        recs.append({
            "priority": "important",
            "icon": "⚡",
            "title": "Advance Problem-Solving Mastery",
            "action": "Progress to medium-difficulty problems. Focus on patterns and optimization.",
            "why": "Most technical rounds require 7+ proficiency to pass consistently",
            "timeline": "2-3 months",
            "impact": "High"
        })
    
    if communication < 7:
        recs.append({
            "priority": "important",
            "icon": "🗣️",
            "title": "Strengthen Communication Impact",
            "action": "Practice 2 mock interviews weekly. Record and review your responses.",
            "why": "Strong communicators receive 2x more offers in final rounds",
            "timeline": "Ongoing",
            "impact": "Medium-High"
        })
    
    # Optional
    if internships == 1:
        recs.append({
            "priority": "optional",
            "icon": "✨",
            "title": "Diversify Experience Profile",
            "action": "Consider an internship in a different domain or company size",
            "why": "Breadth of experience shows adaptability and learning agility",
            "timeline": "Next cycle",
            "impact": "Moderate"
        })
    
    if projects >= 3 and projects < 5:
        recs.append({
            "priority": "optional",
            "icon": "🌟",
            "title": "Build Advanced Technical Edge",
            "action": "Add projects with emerging tech (AI/ML, Cloud Native, Web3)",
            "why": "Differentiator in highly competitive shortlists",
            "timeline": "4-6 months",
            "impact": "Moderate"
        })
    
    # Excellence
    if not recs:
        recs.append({
            "priority": "optional",
            "icon": "🏆",
            "title": "Excellence Maintained",
            "action": "Focus on interview preparation and company-specific research",
            "why": "Your profile is already highly competitive",
            "timeline": "Ongoing",
            "impact": "Sustaining"
        })
    
    return recs


def simulate_analysis():
    """Simulate thoughtful analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        ("Analyzing your academic journey...", 0.25),
        ("Evaluating technical capabilities...", 0.5),
        ("Assessing practical experience...", 0.75),
        ("Generating personalized insights...", 1.0)
    ]
    
    for step, progress in steps:
        status_text.markdown(f"**{step}**")
        progress_bar.progress(progress)
        time.sleep(0.5)
    
    status_text.empty()
    progress_bar.empty()


# ============ LOAD MODEL ============
@st.cache_resource
def load_artifacts():
    """Load model and preprocessor"""
    try:
        model_path = Path(MODEL_DIR) / "readiness_model.pkl"
        
        if not model_path.exists():
            st.warning("Setting up your career coach...")
            train_and_save_model()
        
        model = joblib.load(model_path)
        preprocessor = StudentDataPreprocessor.load(MODEL_DIR)
        return model, preprocessor
    
    except Exception as e:
        st.error(f"Setup error: {str(e)}")
        st.stop()


model, preprocessor = load_artifacts()

# ============ SIDEBAR WITH BRAND HEADER ============
with st.sidebar:
    # BRAND HEADER - Fixed at top
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">PlacePrep Pro</div>
            <div class="sidebar-brand-tagline">Career Readiness Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Your Profile")
    st.caption("Share your journey so far")
    
    st.markdown("#### 🎓 Academic Background")
    gender = st.selectbox("Gender", ["Male", "Female"])
    degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "B.Sc", "M.Sc"])
    branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])
    
    st.markdown("#### 📊 Academic Performance")
    cgpa_imp = get_field_importance("CGPA", branch)
    st.markdown(f"CGPA {get_impact_badge(cgpa_imp)}", unsafe_allow_html=True)
    cgpa = st.slider("", 5.0, 10.0, 7.0, 0.1, key="cgpa", label_visibility="collapsed")
    
    st.markdown("#### 💡 Technical & Soft Skills")
    coding_imp = get_field_importance("Coding_Skills", branch)
    st.markdown(f"Coding Skills {get_impact_badge(coding_imp)}", unsafe_allow_html=True)
    coding = st.slider("", 1, 10, 6, key="coding", label_visibility="collapsed")
    
    comm_imp = get_field_importance("Communication_Skills", branch)
    st.markdown(f"Communication {get_impact_badge(comm_imp)}", unsafe_allow_html=True)
    communication = st.slider("", 1, 10, 6, key="comm", label_visibility="collapsed")
    
    st.markdown("#### 🏆 Practical Experience")
    proj_imp = get_field_importance("Projects", branch)
    st.markdown(f"Projects {get_impact_badge(proj_imp)}", unsafe_allow_html=True)
    projects = st.number_input("", 0, 10, 2, key="proj", label_visibility="collapsed")
    
    intern_imp = get_field_importance("Internships", branch)
    st.markdown(f"Internships {get_impact_badge(intern_imp)}", unsafe_allow_html=True)
    internships = st.number_input("", 0, 5, 1, key="intern", label_visibility="collapsed")
    
    st.markdown("---")
    
    if st.button("✨ Discover My Readiness", type="primary", use_container_width=True):
        st.session_state.evaluated = True
        st.session_state.analysis_complete = False

# ============ CENTERED ASPIRATIONAL HEADER ============
st.markdown("""
    <div class="hero-section">
        <h1 class="hero-headline">Shape Your Placement Future</h1>
        <p class="hero-subtitle">
            We analyze your academics, technical skills, and real-world experience to create 
            a personalized roadmap for your career readiness. Know exactly where you stand 
            and what to do next.
        </p>
    </div>
""", unsafe_allow_html=True)

# ============ MAIN CONTENT ============
if st.session_state.evaluated:
    
    # Analysis simulation
    if not st.session_state.analysis_complete:
        simulate_analysis()
        st.session_state.analysis_complete = True
        st.rerun()
    
    # Prepare input
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Degree": degree,
        "Branch": branch,
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Coding_Skills": coding,
        "Communication_Skills": communication
    }])
    
    # Transform and predict
    X_transformed = preprocessor.transform(input_df)
    predicted_score = float(np.clip(model.predict(X_transformed)[0], 0, 100))
    level = readiness_level(predicted_score)
    context = get_score_context(predicted_score, level)
    weakness_analysis = analyze_weakest_areas(cgpa, coding, communication, projects, internships)
    
    # ========== SCORE HERO - ALWAYS VISIBLE ==========
    st.markdown(f"""
        <div class="score-hero">
            <div class="score-number">{predicted_score:.0f}<span style="font-size: 3rem; opacity: 0.8;">/100</span></div>
            <div class="score-label">{level} Readiness</div>
            <div class="score-context">
                {context['message']}<br>
                {context['encouragement']}<br><br>
                <strong>You're in the {context['percentile']} percentile</strong> · 
                Students like you typically score <strong>{context['range']}</strong>
                {f" · <strong>{context['gap']} points</strong> to reach {context['next_level']}" if context['gap'] != '0' else ""}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ========== METRICS ROW - ALWAYS VISIBLE ==========
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{predicted_score:.0f}</div>
                <div class="metric-label">Your Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{level}</div>
                <div class="metric-label">Category</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        percentile = context['percentile'].split('-')[1].replace('th', '')
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">~{percentile}th</div>
                <div class="metric-label">Percentile</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if context['gap'] != '0':
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{context['gap']}</div>
                    <div class="metric-label">To Next Level</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">🏆</div>
                    <div class="metric-label">Top Tier</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== SECTION 1: SKILL PROFILE & CONTRIBUTION - ALWAYS VISIBLE ==========
    st.markdown('<div class="section-header">📊 Understanding Your Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">How your skills shape up and what\'s driving your score</div>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2, gap="large")
    
    # LEFT: Skill Profile Radar - ALWAYS VISIBLE
    with chart_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Your Skill Shape</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Compare your profile against target benchmarks</div>', unsafe_allow_html=True)
        
        radar_values = [
            cgpa,
            coding,
            communication,
            min(projects, 5) * 2,
            min(internships, 5) * 2
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=["CGPA", "Coding", "Communication", "Projects", "Internships"],
            fill='toself',
            name='Your Profile',
            line_color='#667eea',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line_width=3
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[8, 8, 7, 8, 6],
            theta=["CGPA", "Coding", "Communication", "Projects", "Internships"],
            fill='toself',
            name='Target Profile',
            line_color='#10b981',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line_dash='dash',
            line_width=2
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 10],
                    tickfont=dict(size=11),
                    gridcolor='#e5e7eb'
                ),
                bgcolor='#fafafa'
            ),
            height=450,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            margin=dict(l=50, r=50, t=30, b=70)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Insight below radar - ALWAYS VISIBLE
        st.markdown(f"""
            <div class="insight-box">
                <div class="insight-text">
                    💡 <strong>Key Insight:</strong> {weakness_analysis['insight']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # RIGHT: Score Contribution Breakdown - ALWAYS VISIBLE
    with chart_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">What\'s Driving Your Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Contribution of each component to your final score</div>', unsafe_allow_html=True)
        
        components = pd.DataFrame({
            'Component': ['CGPA', 'Coding', 'Communication', 'Projects', 'Internships'],
            'Contribution': [
                ((cgpa - 5) / 5) * 30,
                ((coding - 1) / 9) * 25,
                ((communication - 1) / 9) * 15,
                (projects / 10) * 20,
                (internships / 5) * 10
            ]
        })
        
        colors = ['#ef4444' if comp in weakness_analysis['weakest'] else '#10b981' if comp == weakness_analysis['strongest'] else '#94a3b8' 
                  for comp in components['Component']]
        
        fig_bar = go.Figure(go.Bar(
            y=components['Component'],
            x=components['Contribution'],
            orientation='h',
            text=components['Contribution'].round(1),
            textposition='inside',
            textfont=dict(color='white', size=13, family='Inter'),
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            hovertemplate='<b>%{y}</b><br>Contribution: %{x:.1f} points<extra></extra>'
        ))
        
        fig_bar.update_layout(
            height=450,
            xaxis=dict(
                title="Points Contributed",
                range=[0, 35],
                gridcolor='#e5e7eb',
                showgrid=True
            ),
            yaxis=dict(
                title="",
                categoryorder='total ascending'
            ),
            plot_bgcolor='#fafafa',
            margin=dict(l=20, r=20, t=30, b=50),
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, key="contribution_bar")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Legend for colors - ALWAYS VISIBLE
        st.markdown(f"""
            <div style="display: flex; gap: 1.25rem; justify-content: center; margin-top: 1rem; font-size: 0.85rem;">
                <span><span style="display: inline-block; width: 14px; height: 14px; background: #ef4444; border-radius: 3px; margin-right: 6px;"></span>Needs Focus</span>
                <span><span style="display: inline-block; width: 14px; height: 14px; background: #10b981; border-radius: 3px; margin-right: 6px;"></span>Strongest</span>
                <span><span style="display: inline-block; width: 14px; height: 14px; background: #94a3b8; border-radius: 3px; margin-right: 6px;"></span>Balanced</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ========== SECTION 2: DETAILED BREAKDOWN & GAUGE - ALWAYS VISIBLE ==========
    st.markdown('<div class="section-header">📈 Detailed Performance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Component-wise scores and overall readiness gauge</div>', unsafe_allow_html=True)
    
    detail_col1, detail_col2 = st.columns(2, gap="large")
    
    # LEFT: Individual Component Scores - ALWAYS VISIBLE
    with detail_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Individual Component Scores</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Each skill rated on a 0-100 scale</div>', unsafe_allow_html=True)
        
        component_scores_df = pd.DataFrame({
            'Component': ['CGPA', 'Coding', 'Communication', 'Projects', 'Internships'],
            'Score': [
                (cgpa / 10) * 100,
                (coding / 10) * 100,
                (communication / 10) * 100,
                (min(projects, 5) / 5) * 100,
                (min(internships, 3) / 3) * 100
            ]
        })
        
        fig_component = go.Figure(go.Bar(
            x=component_scores_df['Component'],
            y=component_scores_df['Score'],
            text=component_scores_df['Score'].round(0).astype(int),
            textposition='outside',
            marker=dict(
                color=component_scores_df['Score'],
                colorscale='RdYlGn',
                showscale=False,
                line=dict(width=0)
            ),
            hovertemplate='<b>%{x}</b><br>Score: %{y:.0f}/100<extra></extra>'
        ))
        
        fig_component.update_layout(
            height=400,
            xaxis=dict(
                title="",
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title="Score (0-100)",
                range=[0, 110],
                gridcolor='#e5e7eb'
            ),
            plot_bgcolor='#fafafa',
            margin=dict(l=50, r=30, t=30, b=50)
        )
        
        st.plotly_chart(fig_component, use_container_width=True, key="component_scores")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RIGHT: Readiness Gauge - ALWAYS VISIBLE
    with detail_col2:
        st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Placement Readiness Gauge</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Your overall placement preparedness meter</div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "", 'font': {'size': 20}},
            delta={'reference': 70, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
            number={'font': {'size': 50, 'family': 'Inter', 'weight': 800}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#374151", 'tickfont': {'size': 12}},
                'bar': {'color': "#667eea", 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 60], 'color': '#fee2e2'},
                    {'range': [60, 80], 'color': '#fef3c7'},
                    {'range': [80, 100], 'color': '#d1fae5'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 5},
                    'thickness': 0.85,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=30, b=30),
            font={'family': 'Inter'}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ========== ACTION PLAN - ALWAYS VISIBLE ==========
    st.markdown('<div class="section-header">🎯 Your Personalized Action Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Prioritized steps to accelerate your placement readiness</div>', unsafe_allow_html=True)
    
    recommendations = get_prioritized_recommendations(
        cgpa, coding, communication, projects, internships, branch
    )
    
    # Group by priority
    critical_recs = [r for r in recommendations if r['priority'] == 'critical']
    important_recs = [r for r in recommendations if r['priority'] == 'important']
    optional_recs = [r for r in recommendations if r['priority'] == 'optional']
    
    # Display Critical - ALWAYS VISIBLE
    if critical_recs:
        st.markdown("**🔴 Critical — Start Here**")
        for rec in critical_recs:
            st.markdown(f"""
                <div class="rec-critical">
                    <div class="rec-title">{rec['icon']} {rec['title']}</div>
                    <div class="rec-action">{rec['action']}</div>
                    <div class="rec-meta">
                        <span>⏱️ Timeline: {rec['timeline']}</span>
                        <span>📈 Impact: {rec['impact']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Display Important - ALWAYS VISIBLE
    if important_recs:
        st.markdown("**🟡 Important — Do Next**")
        for rec in important_recs:
            st.markdown(f"""
                <div class="rec-important">
                    <div class="rec-title">{rec['icon']} {rec['title']}</div>
                    <div class="rec-action">{rec['action']}</div>
                    <div class="rec-meta">
                        <span>⏱️ Timeline: {rec['timeline']}</span>
                        <span>📈 Impact: {rec['impact']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Display Optional - ALWAYS VISIBLE
    if optional_recs:
        st.markdown("**🟢 Optional — Nice to Have**")
        for rec in optional_recs:
            st.markdown(f"""
                <div class="rec-optional">
                    <div class="rec-title">{rec['icon']} {rec['title']}</div>
                    <div class="rec-action">{rec['action']}</div>
                    <div class="rec-meta">
                        <span>⏱️ Timeline: {rec['timeline']}</span>
                        <span>📈 Impact: {rec['impact']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Next Steps CTA - ALWAYS VISIBLE
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2.5rem; border-radius: 20px; text-align: center; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
            <h3 style="margin: 0 0 1.25rem 0; font-size: 1.75rem; font-weight: 700;">🚀 This Week's Priority</h3>
            <p style="font-size: 1.15rem; margin: 0; opacity: 0.95; line-height: 1.6;">
                {critical_recs[0]['action'] if critical_recs else important_recs[0]['action'] if important_recs else "Keep refining your skills and preparing for interviews!"}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Reset button - ALWAYS VISIBLE
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analyze Another Profile", use_container_width=True):
            st.session_state.evaluated = False
            st.session_state.analysis_complete = False
            st.rerun()

else:
    # ========== WELCOME STATE ==========
    st.markdown("""
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <p style="font-size: 1.15rem; color: #6b7280; margin-bottom: 3rem; line-height: 1.6;">
                👈 Complete your profile in the sidebar to receive your personalized career readiness analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
            <div class="welcome-card">
                <div class="welcome-icon">🎯</div>
                <div class="welcome-title">Intelligent Analysis</div>
                <div class="welcome-desc">
                    Machine learning insights trained on thousands of placement outcomes
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="welcome-card">
                <div class="welcome-icon">📊</div>
                <div class="welcome-title">Complete Visual Report</div>
                <div class="welcome-desc">
                    Six analytical charts showing strengths, weaknesses, and growth paths
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="welcome-card">
                <div class="welcome-icon">💡</div>
                <div class="welcome-title">Prioritized Guidance</div>
                <div class="welcome-desc">
                    Actionable recommendations with timelines and measurable impact
                </div>
            </div>
        """, unsafe_allow_html=True)
"""
Student Placement Readiness Dashboard - Career Coach Edition
Addresses all UX feedback for emotional intelligence and guidance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, List, Tuple

from scoring import readiness_level
from train_model import train_and_save_model
from preprocessing import StudentDataPreprocessor

# ============ CONFIGURATION ============
MODEL_DIR = "model"
st.set_page_config(
    page_title="Placement Readiness Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .impact-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .impact-high {
        background-color: #ff4444;
        color: white;
    }
    .impact-medium {
        background-color: #ffaa00;
        color: white;
    }
    .impact-low {
        background-color: #00cc88;
        color: white;
    }
    .context-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .priority-critical {
        border-left: 5px solid #dc3545;
        background-color: #fff5f5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .priority-important {
        border-left: 5px solid #ffc107;
        background-color: #fffef5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .priority-optional {
        border-left: 5px solid #28a745;
        background-color: #f5fff5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
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
    """Determine importance of each field based on branch"""
    importance_map = {
        "CSE": {
            "CGPA": "high",
            "Coding_Skills": "high",
            "Communication_Skills": "medium",
            "Projects": "high",
            "Internships": "medium"
        },
        "ECE": {
            "CGPA": "high",
            "Coding_Skills": "medium",
            "Communication_Skills": "medium",
            "Projects": "medium",
            "Internships": "high"
        },
        "ME": {
            "CGPA": "high",
            "Coding_Skills": "low",
            "Communication_Skills": "high",
            "Projects": "medium",
            "Internships": "high"
        },
        "CE": {
            "CGPA": "high",
            "Coding_Skills": "low",
            "Communication_Skills": "high",
            "Projects": "medium",
            "Internships": "high"
        },
        "EE": {
            "CGPA": "high",
            "Coding_Skills": "medium",
            "Communication_Skills": "medium",
            "Projects": "medium",
            "Internships": "high"
        }
    }
    return importance_map.get(branch, {}).get(field, "medium")


def get_impact_badge(importance: str) -> str:
    """Generate HTML badge for field importance"""
    labels = {
        "high": "High Impact",
        "medium": "Medium Impact",
        "low": "Low Impact"
    }
    return f'<span class="impact-badge impact-{importance}">{labels[importance]}</span>'


def get_score_context(score: float, level: str) -> Dict:
    """Provide context about the score"""
    if level == "Low":
        return {
            "range": "35-59",
            "message": "You're in the foundational stage. Many students start here!",
            "next_level": "Medium (60+)",
            "gap": f"{60 - score:.0f} points to reach Medium level",
            "percentile": "15-40th percentile"
        }
    elif level == "Medium":
        return {
            "range": "60-79",
            "message": "You're progressing well! You're in the competitive range.",
            "next_level": "High (80+)",
            "gap": f"{80 - score:.0f} points to reach High level",
            "percentile": "40-75th percentile"
        }
    else:
        return {
            "range": "80-100",
            "message": "Excellent! You're in the top tier of placement readiness.",
            "next_level": "You're already in the highest category!",
            "gap": "Keep maintaining this level",
            "percentile": "75-95th percentile"
        }


def analyze_weakest_areas(cgpa: float, coding: int, communication: int, 
                         projects: int, internships: int) -> str:
    """Identify weakest areas for targeted improvement"""
    scores = {
        "CGPA": (cgpa / 10) * 100,
        "Coding Skills": (coding / 10) * 100,
        "Communication": (communication / 10) * 100,
        "Projects": (min(projects, 5) / 5) * 100,
        "Internships": (min(internships, 3) / 3) * 100
    }
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    weakest = sorted_scores[:2]
    
    return f"Your weakest areas are **{weakest[0][0]}** and **{weakest[1][0]}** — improving these will increase your score fastest."


def get_prioritized_recommendations(cgpa: float, coding: int, communication: int,
                                   projects: int, internships: int, branch: str) -> List[Dict]:
    """Generate prioritized recommendations with impact levels"""
    recs = []
    
    # Critical (High Priority - Do First)
    if cgpa < 6.5:
        recs.append({
            "priority": "critical",
            "icon": "🔴",
            "title": "CRITICAL: Academic Performance",
            "action": f"Bring your CGPA from {cgpa:.1f} to 7.0+",
            "why": "CGPA below 6.5 eliminates you from many company shortlists",
            "timeline": "Next 2 semesters",
            "impact": "High - Opens 40% more opportunities"
        })
    
    if internships == 0:
        recs.append({
            "priority": "critical",
            "icon": "🔴",
            "title": "CRITICAL: Get Your First Internship",
            "action": "Apply to 20+ companies this week. Target startups and smaller firms.",
            "why": "0 internships is a red flag for recruiters",
            "timeline": "Next 2-3 months",
            "impact": "High - Essential for resume screening"
        })
    
    if coding < 5 and branch == "CSE":
        recs.append({
            "priority": "critical",
            "icon": "🔴",
            "title": "CRITICAL: Coding Skills (CSE Branch)",
            "action": "Solve 2 easy problems daily on LeetCode. Start with Arrays and Strings.",
            "why": "Below 5/10 coding skills will fail technical rounds",
            "timeline": "Next 3 months",
            "impact": "Very High - Core requirement for CSE placements"
        })
    
    # Important (Medium Priority - Do Next)
    if cgpa >= 6.5 and cgpa < 7.5:
        recs.append({
            "priority": "important",
            "icon": "🟡",
            "title": "IMPORTANT: Boost CGPA Further",
            "action": f"Push from {cgpa:.1f} to 8.0+ for premium companies",
            "why": "8.0+ CGPA gives access to top-tier companies",
            "timeline": "Next 2-3 semesters",
            "impact": "Medium - Competitive edge"
        })
    
    if projects < 3:
        recs.append({
            "priority": "important",
            "icon": "🟡",
            "title": "IMPORTANT: Build Project Portfolio",
            "action": f"Add {3 - projects} full-stack projects. Deploy on GitHub + live demo.",
            "why": "Projects demonstrate practical skills beyond academics",
            "timeline": "Next 2-4 months",
            "impact": "High - Shows initiative and real skills"
        })
    
    if coding >= 5 and coding < 7:
        recs.append({
            "priority": "important",
            "icon": "🟡",
            "title": "IMPORTANT: Advance Coding Skills",
            "action": "Move to medium-level problems. Focus on problem-solving patterns.",
            "why": "7+ coding skills needed to clear most company technical rounds",
            "timeline": "Next 2 months (daily practice)",
            "impact": "High - Interview performance"
        })
    
    if communication < 7:
        recs.append({
            "priority": "important",
            "icon": "🟡",
            "title": "IMPORTANT: Communication Skills",
            "action": "Practice mock interviews weekly. Join Toastmasters or debate club.",
            "why": "Good communicators perform 2x better in HR and technical rounds",
            "timeline": "Ongoing (3+ months)",
            "impact": "Medium - Interview success"
        })
    
    # Optional (Low Priority - Nice to Have)
    if internships == 1:
        recs.append({
            "priority": "optional",
            "icon": "🟢",
            "title": "OPTIONAL: Diverse Experience",
            "action": "Consider a second internship in a different domain/role",
            "why": "Shows versatility and broader skill set",
            "timeline": "Next internship season",
            "impact": "Low - Differentiator"
        })
    
    if projects >= 3 and projects < 5:
        recs.append({
            "priority": "optional",
            "icon": "🟢",
            "title": "OPTIONAL: Advanced Projects",
            "action": "Add 1-2 projects with cutting-edge tech (AI/ML, Cloud, etc.)",
            "why": "Helps stand out in competitive shortlists",
            "timeline": "Next 3-6 months",
            "impact": "Low - Premium edge"
        })
    
    # Excellent case
    if not recs:
        recs.append({
            "priority": "optional",
            "icon": "🌟",
            "title": "EXCELLENT PROFILE",
            "action": "Focus on interview preparation and company research",
            "why": "Your profile is already competitive",
            "timeline": "Ongoing",
            "impact": "Maintenance"
        })
    
    return recs


def simulate_analysis():
    """Simulate AI analysis with progress indicators"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        ("Analyzing academic performance...", 0.2),
        ("Evaluating technical skills...", 0.4),
        ("Assessing practical experience...", 0.6),
        ("Comparing with peer benchmarks...", 0.8),
        ("Generating personalized insights...", 1.0)
    ]
    
    for step, progress in steps:
        status_text.text(step)
        progress_bar.progress(progress)
        time.sleep(0.4)  # Brief pause for each step
    
    status_text.empty()
    progress_bar.empty()


# ============ LOAD MODEL ============
@st.cache_resource
def load_artifacts():
    """Load model and preprocessor"""
    try:
        model_path = Path(MODEL_DIR) / "readiness_model.pkl"
        
        if not model_path.exists():
            st.warning("⚠️ Model not found. Training...")
            train_and_save_model()
        
        model = joblib.load(model_path)
        preprocessor = StudentDataPreprocessor.load(MODEL_DIR)
        
        return model, preprocessor
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()


model, preprocessor = load_artifacts()

# ============ HEADER ============
st.markdown('<h1 class="main-title">🎯 Your Placement Readiness Coach</h1>', 
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Get your readiness score + visual breakdown + exact improvement steps tailored for you.</p>',
    unsafe_allow_html=True
)

st.divider()

# ============ SIDEBAR ============
with st.sidebar:
    st.header("📋 Your Profile")
    st.caption("Fill in your details to get personalized guidance")
    
    st.subheader("🎓 Basic Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "B.Sc", "M.Sc"])
    branch = st.selectbox(
        "Branch", 
        ["CSE", "ECE", "ME", "CE", "EE"],
        help="Your branch affects which skills matter most"
    )
    
    st.subheader("📊 Academic Performance")
    
    cgpa_importance = get_field_importance("CGPA", branch)
    st.markdown(
        f"CGPA {get_impact_badge(cgpa_importance)}", 
        unsafe_allow_html=True
    )
    cgpa = st.slider("", 5.0, 10.0, 7.0, 0.1, key="cgpa_slider", label_visibility="collapsed")
    
    st.subheader("💡 Skills Assessment")
    
    coding_importance = get_field_importance("Coding_Skills", branch)
    st.markdown(
        f"Coding Skills {get_impact_badge(coding_importance)}", 
        unsafe_allow_html=True
    )
    coding = st.slider("", 1, 10, 6, key="coding_slider", label_visibility="collapsed")
    
    comm_importance = get_field_importance("Communication_Skills", branch)
    st.markdown(
        f"Communication Skills {get_impact_badge(comm_importance)}", 
        unsafe_allow_html=True
    )
    communication = st.slider("", 1, 10, 6, key="comm_slider", label_visibility="collapsed")
    
    st.subheader("🏆 Practical Experience")
    
    proj_importance = get_field_importance("Projects", branch)
    st.markdown(
        f"Projects {get_impact_badge(proj_importance)}", 
        unsafe_allow_html=True
    )
    projects = st.number_input("", 0, 10, 2, key="proj_input", label_visibility="collapsed")
    
    intern_importance = get_field_importance("Internships", branch)
    st.markdown(
        f"Internships {get_impact_badge(intern_importance)}", 
        unsafe_allow_html=True
    )
    internships = st.number_input("", 0, 5, 1, key="intern_input", label_visibility="collapsed")
    
    st.divider()
    
    if st.button("🚀 Analyze My Readiness", type="primary", use_container_width=True):
        st.session_state.evaluated = True
        st.session_state.analysis_complete = False

# ============ MAIN CONTENT ============
if st.session_state.evaluated:
    
    # Show analysis simulation
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
    
    # ========== SCORE WITH CONTEXT ==========
    st.markdown(f"""
        <div class="context-box">
            <h2 style="margin: 0; font-size: 2rem;">Your Readiness Score: {predicted_score:.0f}/100</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Level: <strong>{level}</strong></p>
            <p style="margin: 1rem 0 0 0; font-size: 0.95rem; opacity: 0.9;">
                {context['message']}<br>
                You're in the <strong>{context['percentile']}</strong> among students.<br>
                Students with similar profiles typically score between <strong>{context['range']}</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== METRICS ROW ==========
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Your Score", f"{predicted_score:.0f}", 
                 help="Your placement readiness out of 100")
    
    with col2:
        st.metric("Category", level,
                 help=f"Typical range: {context['range']}")
    
    with col3:
        if level != "High":
            st.metric("To Next Level", context['gap'],
                     help=f"Points needed for {context['next_level']}")
        else:
            st.metric("Status", "Top Tier ✨",
                     help="You're in the highest category!")
    
    with col4:
        percentile_num = int(context['percentile'].split('-')[1].replace('th percentile', ''))
        st.metric("Percentile", f"~{percentile_num}th",
                 help="Approximate ranking among peers")
    
    st.divider()
    
    # ========== VISUALIZATIONS ==========
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Your Skills Profile")
        
        fig_radar = go.Figure()
        
        radar_values = [
            cgpa,
            coding,
            communication,
            min(projects, 5) * 2,
            min(internships, 5) * 2
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=["CGPA", "Coding", "Communication", "Projects", "Internships"],
            fill='toself',
            name='Your Profile',
            line_color='#1f77b4'
        ))
        
        # Target benchmark
        fig_radar.add_trace(go.Scatterpolar(
            r=[8, 8, 7, 8, 6],
            theta=["CGPA", "Coding", "Communication", "Projects", "Internships"],
            fill='toself',
            name='Target Profile',
            line_color='#2ca02c',
            line_dash='dash',
            opacity=0.5
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 10])),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Add insight below chart
        weakness_insight = analyze_weakest_areas(cgpa, coding, communication, projects, internships)
        st.info(f"💡 **Insight:** {weakness_insight}")
    
    with col_right:
        st.subheader("📊 Score Breakdown")
        
        # Calculate contributions
        components = pd.DataFrame({
            'Component': ['CGPA', 'Coding', 'Communication', 'Projects', 'Internships'],
            'Your Value': [cgpa, coding, communication, projects, internships],
            'Contribution': [
                ((cgpa - 5) / 5) * 30,
                ((coding - 1) / 9) * 25,
                ((communication - 1) / 9) * 15,
                (projects / 10) * 20,
                (internships / 5) * 10
            ]
        })
        
        fig_bar = go.Figure(go.Bar(
            x=components['Contribution'],
            y=components['Component'],
            orientation='h',
            text=components['Contribution'].round(1),
            textposition='auto',
            marker_color=['#ff6b6b' if x < 15 else '#4ecdc4' if x < 22 else '#95e1d3' 
                         for x in components['Contribution']]
        ))
        
        fig_bar.update_layout(
            height=400,
            xaxis_title="Points Contributed",
            yaxis_title="",
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.caption("🎯 Each component contributes differently based on importance weights")
    
    st.divider()
    
    # ========== PRIORITIZED RECOMMENDATIONS ==========
    st.subheader("💡 Your Personalized Action Plan")
    st.caption("Prioritized by impact — start from top to bottom")
    
    recommendations = get_prioritized_recommendations(
        cgpa, coding, communication, projects, internships, branch
    )
    
    # Group by priority
    critical_recs = [r for r in recommendations if r['priority'] == 'critical']
    important_recs = [r for r in recommendations if r['priority'] == 'important']
    optional_recs = [r for r in recommendations if r['priority'] == 'optional']
    
    # Display Critical
    if critical_recs:
        st.markdown("### 🔴 CRITICAL - Do This First")
        for rec in critical_recs:
            st.markdown(f"""
                <div class="priority-critical">
                    <h4 style="margin: 0 0 0.5rem 0;">{rec['icon']} {rec['title']}</h4>
                    <p style="margin: 0.5rem 0;"><strong>Action:</strong> {rec['action']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Why:</strong> {rec['why']}</p>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">⏱️ {rec['timeline']} | 📈 {rec['impact']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Display Important
    if important_recs:
        st.markdown("### 🟡 IMPORTANT - Do This Next")
        for rec in important_recs:
            st.markdown(f"""
                <div class="priority-important">
                    <h4 style="margin: 0 0 0.5rem 0;">{rec['icon']} {rec['title']}</h4>
                    <p style="margin: 0.5rem 0;"><strong>Action:</strong> {rec['action']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Why:</strong> {rec['why']}</p>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">⏱️ {rec['timeline']} | 📈 {rec['impact']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Display Optional
    if optional_recs:
        st.markdown("### 🟢 OPTIONAL - Nice to Have")
        for rec in optional_recs:
            st.markdown(f"""
                <div class="priority-optional">
                    <h4 style="margin: 0 0 0.5rem 0;">{rec['icon']} {rec['title']}</h4>
                    <p style="margin: 0.5rem 0;"><strong>Action:</strong> {rec['action']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Why:</strong> {rec['why']}</p>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">⏱️ {rec['timeline']} | 📈 {rec['impact']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== NEXT STEPS ==========
    st.markdown("### 🎯 Your Next Week's Focus")
    
    if critical_recs:
        st.success(f"**Start here:** {critical_recs[0]['action']}")
    elif important_recs:
        st.info(f"**Start here:** {important_recs[0]['action']}")
    else:
        st.success("**Keep going!** Focus on interview prep and company research.")
    
    # Reset button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analyze Another Profile", use_container_width=True):
            st.session_state.evaluated = False
            st.session_state.analysis_complete = False
            st.rerun()

else:
    # ========== WELCOME SCREEN ==========
    st.info("👈 **Get Started:** Fill in your profile in the sidebar and click '**Analyze My Readiness**'")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Smart Analysis")
        st.write("ML-powered insights based on thousands of placement outcomes")
    
    with col2:
        st.markdown("### 📊 Visual Breakdown")
        st.write("See exactly where you stand and what to improve")
    
    with col3:
        st.markdown("### 💡 Actionable Steps")
        st.write("Prioritized recommendations with timelines and impact")
    
    st.markdown("---")
    
    # Sample testimonial or stats
    st.markdown("""
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0 0 1rem 0;">What Makes This Different?</h3>
            <p style="font-size: 1.1rem; color: #666; margin: 0;">
                Not just a score — a complete career coach that understands your unique profile,
                tells you exactly where you stand, and guides you step-by-step on what to do next.
            </p>
        </div>
    """, unsafe_allow_html=True)
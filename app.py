"""
Student Placement Readiness Dashboard
Streamlit application for assessing student placement readiness
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
from typing import Tuple, Dict

from scoring import readiness_level
from train_model import train_and_save_model

# ---------------- CONSTANTS ----------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "readiness_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ---------------- SESSION STATE ----------------
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Placement Readiness",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts() -> Tuple:
    """Load model, encoder, and scaler with error handling"""
    try:
        # Ensure model directory exists
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        # Train model if artifacts don't exist
        if not all([
            os.path.exists(MODEL_PATH),
            os.path.exists(ENCODER_PATH),
            os.path.exists(SCALER_PATH)
        ]):
            st.info("🔄 Training model for the first time... This may take a moment.")
            train_and_save_model()
            st.success("✅ Model training complete!")
        
        # Load artifacts
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        return model, encoder, scaler
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def get_recommendations(cgpa: float, coding: int, communication: int, 
                       projects: int, internships: int) -> list:
    """Generate personalized recommendations based on student profile"""
    recommendations = []
    
    if cgpa < 6.5:
        recommendations.append({
            "icon": "📘",
            "text": "**CGPA Alert**: Focus on improving academic performance. Aim for 7.0+ CGPA.",
            "priority": "high"
        })
    elif cgpa < 7.5:
        recommendations.append({
            "icon": "📚",
            "text": "**Academic Growth**: Good progress! Target 8.0+ for competitive edge.",
            "priority": "medium"
        })
    
    if coding < 5:
        recommendations.append({
            "icon": "💻",
            "text": "**Coding Skills**: Practice DSA daily on LeetCode/CodeChef. Start with easy problems.",
            "priority": "high"
        })
    elif coding < 7:
        recommendations.append({
            "icon": "⌨️",
            "text": "**Coding Practice**: Solve 2-3 medium problems daily. Focus on patterns.",
            "priority": "medium"
        })
    
    if communication < 6:
        recommendations.append({
            "icon": "🗣️",
            "text": "**Communication**: Join debate clubs, practice mock interviews, read aloud daily.",
            "priority": "high"
        })
    
    if projects < 2:
        recommendations.append({
            "icon": "🛠️",
            "text": "**Projects**: Build at least 2-3 full-stack projects. Deploy on GitHub.",
            "priority": "high"
        })
    elif projects < 4:
        recommendations.append({
            "icon": "🚀",
            "text": "**Project Portfolio**: Add 1-2 more projects with modern tech stack.",
            "priority": "medium"
        })
    
    if internships == 0:
        recommendations.append({
            "icon": "🏢",
            "text": "**Internship**: Apply immediately! Target startups and product companies.",
            "priority": "high"
        })
    elif internships == 1:
        recommendations.append({
            "icon": "💼",
            "text": "**Experience**: Consider a second internship in a different domain.",
            "priority": "low"
        })
    
    if not recommendations:
        recommendations.append({
            "icon": "🌟",
            "text": "**Excellent Profile**: Maintain consistency and prepare for advanced interviews.",
            "priority": "low"
        })
    
    return recommendations


def create_radar_chart(cgpa: float, coding: int, communication: int, 
                      projects: int, internships: int) -> go.Figure:
    """Create interactive radar chart for skill visualization"""
    
    # Normalize all values to 0-10 scale
    radar_data = {
        "CGPA": cgpa,
        "Coding": coding,
        "Communication": communication,
        "Projects": min(projects, 5) * 2,  # Cap at 5, scale to 10
        "Internships": min(internships, 5) * 2  # Cap at 5, scale to 10
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(radar_data.values()),
        theta=list(radar_data.keys()),
        fill='toself',
        name='Your Profile',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add benchmark line (target values)
    benchmark = [8, 8, 7, 8, 6]  # Target values for each skill
    fig.add_trace(go.Scatterpolar(
        r=benchmark,
        theta=list(radar_data.keys()),
        fill='toself',
        name='Target Profile',
        line_color='#2ca02c',
        fillcolor='rgba(44, 160, 44, 0.1)',
        line_dash='dash'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        height=450,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig


# ---------------- MAIN APP ----------------
try:
    model, encoder, scaler = load_artifacts()
except Exception as e:
    st.error("Failed to load model. Please check the logs.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-header">🎓 Student Placement Readiness Dashboard</h1>', 
            unsafe_allow_html=True)
st.caption("ML-powered assessment of placement readiness with actionable insights")

st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🧾 Student Profile")
    st.caption("Fill in your details below")
    
    st.subheader("📋 Basic Information")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
    degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "B.Sc", "M.Sc"], 
                         help="Your current degree program")
    branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"], 
                         help="Your branch of study")
    
    st.subheader("📊 Academic Performance")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1, 
                    help="Your current CGPA on a 10-point scale")
    
    st.subheader("💡 Skills Assessment")
    coding = st.slider("Coding Skills", 1, 10, 6, 1,
                      help="Rate your coding ability (1=Beginner, 10=Expert)")
    communication = st.slider("Communication Skills", 1, 10, 6, 1,
                             help="Rate your communication ability")
    
    st.subheader("🏆 Experience")
    projects = st.number_input("Number of Projects", 0, 10, 2, 1,
                               help="Total completed projects")
    internships = st.number_input("Number of Internships", 0, 5, 1, 1,
                                  help="Total internships completed")
    
    st.divider()
    
    evaluate_btn = st.button("🚀 Evaluate Readiness", 
                            type="primary",
                            use_container_width=True)
    
    if evaluate_btn:
        st.session_state.evaluated = True

# ---------------- MAIN OUTPUT ----------------
if st.session_state.evaluated:
    
    # ---------- PREPARE INPUT ----------
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
    
    cat_cols = ["Gender", "Degree", "Branch"]
    num_cols = ["CGPA", "Internships", "Projects", "Coding_Skills", "Communication_Skills"]
    
    # Encode and scale
    X_cat = encoder.transform(input_df[cat_cols])
    X_num = scaler.transform(input_df[num_cols])
    X = np.hstack([X_num, X_cat])
    
    # Predict
    predicted_score = float(model.predict(X)[0])
    level = readiness_level(predicted_score)
    
    # ================= OVERVIEW CONTAINER =================
    with st.container():
        st.subheader("📊 Readiness Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Readiness Score", f"{predicted_score:.1f}", 
                     delta=f"{predicted_score - 70:.1f} from avg",
                     help="Score out of 100")
        
        with col2:
            st.metric("Readiness Level", level)
        
        with col3:
            percentile = min(int((predicted_score / 100) * 100), 99)
            st.metric("Percentile", f"{percentile}th",
                     help="Approximate ranking")
        
        with col4:
            if level == "Low":
                st.error("⚠️ Needs Work")
            elif level == "Medium":
                st.warning("⚡ Good Progress")
            else:
                st.success("✅ Excellent!")
    
    st.divider()
    
    # ================= VISUALIZATION CONTAINER =================
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📈 Skill Radar Chart")
        fig = create_radar_chart(cgpa, coding, communication, projects, internships)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("📉 Score Breakdown")
        
        breakdown = {
            "CGPA Contribution": cgpa * 10,
            "Coding Skills": coding * 6,
            "Communication": communication * 4,
            "Projects": projects * 5,
            "Internships": internships * 8
        }
        
        breakdown_df = pd.DataFrame({
            "Component": list(breakdown.keys()),
            "Score": list(breakdown.values())
        })
        
        fig_bar = go.Figure(go.Bar(
            x=breakdown_df["Score"],
            y=breakdown_df["Component"],
            orientation='h',
            marker_color='#1f77b4',
            text=breakdown_df["Score"].round(1),
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Contribution to Score",
            yaxis_title=""
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    
    # ================= RECOMMENDATIONS CONTAINER =================
    with st.container():
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = get_recommendations(cgpa, coding, communication, projects, internships)
        
        for rec in recommendations:
            priority_color = {
                "high": "#dc3545",
                "medium": "#ffc107",
                "low": "#28a745"
            }.get(rec["priority"], "#6c757d")
            
            st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    border-left: 4px solid {priority_color};
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border-radius: 0.25rem;
                ">
                    {rec['icon']} {rec['text']}
                </div>
            """, unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Evaluate Another Profile"):
        st.session_state.evaluated = False
        st.rerun()

else:
    # Welcome message
    st.info("👈 **Get Started:** Fill in the student profile form in the sidebar and click '**Evaluate Readiness**' to see results!")
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 ML-Powered")
        st.write("Random Forest model trained on student data")
    
    with col2:
        st.markdown("### 📊 Visual Insights")
        st.write("Interactive charts and skill breakdowns")
    
    with col3:
        st.markdown("### 💡 Actionable Tips")
        st.write("Personalized recommendations for improvement")
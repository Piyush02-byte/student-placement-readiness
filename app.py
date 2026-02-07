"""
Fixed Streamlit app with all visualizations working
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple

from scoring import readiness_level
from train_model import train_and_save_model
from preprocessing import StudentDataPreprocessor

# ============ CONFIGURATION ============
MODEL_DIR = "model"
st.set_page_config(
    page_title="Student Placement Readiness",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ SESSION STATE ============
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

# ============ LOAD MODEL ============
@st.cache_resource
def load_artifacts() -> Tuple:
    """Load model and preprocessor"""
    try:
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        model_path = Path(MODEL_DIR) / "readiness_model.pkl"
        
        if not model_path.exists():
            with st.spinner("🔄 Training model... This may take a moment."):
                train_and_save_model()
        
        model = joblib.load(model_path)
        preprocessor = StudentDataPreprocessor.load(MODEL_DIR)
        
        return model, preprocessor
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


model, preprocessor = load_artifacts()

# ============ HEADER ============
st.title("🎓 Student Placement Readiness Dashboard")
st.caption("ML-powered assessment with comprehensive analytics")
st.divider()

# ============ SIDEBAR ============
with st.sidebar:
    st.header("🧾 Student Profile")
    
    st.subheader("📋 Basic Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "B.Sc", "M.Sc"])
    branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])
    
    st.subheader("📊 Academics")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
    
    st.subheader("💡 Skills")
    coding = st.slider("Coding Skills", 1, 10, 6)
    communication = st.slider("Communication Skills", 1, 10, 6)
    
    st.subheader("🏆 Experience")
    projects = st.number_input("Projects", 0, 10, 2)
    internships = st.number_input("Internships", 0, 5, 1)
    
    st.divider()
    
    if st.button("🚀 Evaluate Readiness", type="primary", use_container_width=True):
        st.session_state.evaluated = True

# ============ MAIN CONTENT ============
if st.session_state.evaluated:
    
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
    
    # ========== OVERVIEW ==========
    st.subheader("📊 Readiness Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Readiness Score", f"{predicted_score:.1f}/100")
    col2.metric("Level", level)
    col3.metric("Percentile", f"~{int((predicted_score/100)*100)}th")
    
    with col4:
        if level == "Low":
            st.error("⚠️ Needs Work")
        elif level == "Medium":
            st.warning("⚡ Good Progress")
        else:
            st.success("✅ Excellent!")
    
    st.divider()
    
    # ========== VISUALIZATIONS ==========
    col_left, col_right = st.columns(2)
    
    # RADAR CHART
    with col_left:
        st.subheader("📈 Skills Radar")
        
        radar_data = {
            "CGPA": cgpa,
            "Coding": coding,
            "Communication": communication,
            "Projects": min(projects, 5) * 2,
            "Internships": min(internships, 5) * 2
        }
        
        fig_radar = go.Figure()
        
        # Actual profile
        fig_radar.add_trace(go.Scatterpolar(
            r=list(radar_data.values()),
            theta=list(radar_data.keys()),
            fill='toself',
            name='Your Profile',
            line_color='#1f77b4'
        ))
        
        # Target profile
        fig_radar.add_trace(go.Scatterpolar(
            r=[8, 8, 7, 8, 6],
            theta=list(radar_data.keys()),
            fill='toself',
            name='Target',
            line_color='#2ca02c',
            line_dash='dash'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 10])),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
    
    # SCORE BREAKDOWN BAR CHART
    with col_right:
        st.subheader("📉 Score Breakdown")
        
        # Calculate actual contributions (normalized)
        MAX_SCORE = 290  # From training
        contributions = {
            "CGPA": (cgpa * 10 / MAX_SCORE) * 100,
            "Coding": (coding * 6 / MAX_SCORE) * 100,
            "Communication": (communication * 4 / MAX_SCORE) * 100,
            "Projects": (min(projects, 10) * 5 / MAX_SCORE) * 100,
            "Internships": (min(internships, 5) * 8 / MAX_SCORE) * 100
        }
        
        fig_bar = go.Figure(go.Bar(
            x=list(contributions.values()),
            y=list(contributions.keys()),
            orientation='h',
            marker_color='#1f77b4',
            text=[f"{v:.1f}" for v in contributions.values()],
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            height=400,
            xaxis_title="Contribution to Score (%)",
            yaxis_title="",
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")
    
    st.divider()
    
    # ========== GAUGE CHART ==========
    st.subheader("🎯 Readiness Gauge")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Placement Readiness", 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffcccc'},
                {'range': [60, 80], 'color': '#ffffcc'},
                {'range': [80, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")
    
    st.divider()
    
    # ========== RECOMMENDATIONS ==========
    st.subheader("💡 Personalized Recommendations")
    
    recs = []
    if cgpa < 7:
        recs.append("📘 **CGPA**: Focus on improving to 7.0+")
    if coding < 7:
        recs.append("💻 **Coding**: Practice DSA daily on LeetCode")
    if communication < 7:
        recs.append("🗣️ **Communication**: Join speaking clubs, practice presentations")
    if projects < 3:
        recs.append("🛠️ **Projects**: Build 2-3 full-stack projects")
    if internships < 1:
        recs.append("🏢 **Internships**: Apply to companies immediately")
    
    if not recs:
        st.success("🌟 **Excellent profile!** Keep preparing for advanced interviews.")
    else:
        for rec in recs:
            st.warning(rec)
    
    # Reset button
    if st.button("🔄 New Evaluation"):
        st.session_state.evaluated = False
        st.rerun()

else:
    st.info("👈 Fill the form and click **Evaluate Readiness**")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown("### 🎯 ML-Powered\nRandom Forest model")
    col2.markdown("### 📊 Visual Insights\nComprehensive charts")
    col3.markdown("### 💡 Smart Tips\nPersonalized guidance")
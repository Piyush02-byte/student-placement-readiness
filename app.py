import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scoring import readiness_level

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Placement Readiness",
    layout="wide"
)

# ---------------- LOAD MODEL & PREPROCESSORS ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/readiness_model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, encoder, scaler


model, encoder, scaler = load_artifacts()

# ---------------- HEADER ----------------
st.title("🎓 Student Placement Readiness Dashboard")
st.caption(
    "Assess a student's placement readiness based on academics, skills, and experience. "
    "Get actionable insights and improvement guidance."
)

st.divider()

# ============================================================
# SIDEBAR — INPUT SECTION
# ============================================================
st.sidebar.header("🧾 Student Profile")

st.sidebar.subheader("🎓 Academic Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
degree = st.sidebar.selectbox("Degree", ["B.Tech", "M.Tech", "B.Sc", "M.Sc"])
branch = st.sidebar.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)

st.sidebar.subheader("💻 Skills")
coding = st.sidebar.slider("Coding Skills (1–10)", 1, 10, 6)
communication = st.sidebar.slider("Communication Skills (1–10)", 1, 10, 6)

st.sidebar.subheader("🧪 Experience")
projects = st.sidebar.number_input("Number of Projects", 0, 10, 2)
internships = st.sidebar.number_input("Number of Internships", 0, 5, 1)

evaluate = st.sidebar.button("🚀 Evaluate Readiness")

# ============================================================
# MAIN AREA — RESULTS & INSIGHTS
# ============================================================
if evaluate:
    # ---------------- PREPARE INPUT ----------------
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

    encoded_cat = encoder.transform(input_df[cat_cols])
    scaled_num = scaler.transform(input_df[num_cols])
    X_input = np.hstack([scaled_num, encoded_cat])

    predicted_score = model.predict(X_input)[0]
    level = readiness_level(predicted_score)

    # ========================================================
    # HERO RESULT SECTION
    # ========================================================
    st.subheader("📊 Readiness Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Placement Readiness Score",
        value=f"{predicted_score:.1f} / 100"
    )

    col2.metric(
        label="Readiness Level",
        value=level
    )

    if level == "Low":
        col3.error("Needs significant improvement")
    elif level == "Medium":
        col3.warning("Moderately prepared")
    else:
        col3.success("Well prepared for placements")

    st.divider()

    # ========================================================
    # INSIGHT SUMMARY (TEXT)
    # ========================================================
    st.subheader("🧠 Summary Insight")

    if level == "Low":
        st.write(
            "Your current profile shows gaps in key areas required for placements. "
            "Improving core skills and gaining experience will significantly boost readiness."
        )
    elif level == "Medium":
        st.write(
            "You have a decent foundation, but targeted improvements in specific areas "
            "can greatly enhance your placement chances."
        )
    else:
        st.write(
            "You demonstrate a strong and balanced profile. "
            "Focus on interview preparation and company-specific practice."
        )

    st.divider()

    # ========================================================
    # ========================================================
# RADAR CHART — SKILL PROFILE
# ========================================================
st.subheader("📈 Skill Profile Overview")

# Normalize values to 0–10 scale for visualization
radar_labels = [
    "CGPA",
    "Coding Skills",
    "Communication Skills",
    "Projects",
    "Internships"
]

radar_values = [
    cgpa,                       # already 0–10
    coding,                     # 1–10
    communication,              # 1–10
    min(projects, 5) * 2,       # scale projects to 0–10
    min(internships, 5) * 2     # scale internships to 0–10
]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=radar_values,
    theta=radar_labels,
    fill='toself',
    name='Student Profile'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )
    ),
    showlegend=False,
    height=450
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Visual Insights (Coming Next)")

st.info(
        "Skill profile charts, feature importance, and comparison graphs "
        "will appear here in the next step."
    )

st.divider()

    # ========================================================
    # RECOMMENDATIONS PLACEHOLDER
    # ========================================================
st.subheader("🧩 Personalized Recommendations")

st.info(
        "Actionable recommendations based on your profile will be displayed here."
    )

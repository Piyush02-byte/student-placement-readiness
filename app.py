import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

from scoring import readiness_level
from train_model import train_and_save_model

# ---------------- SESSION STATE ----------------
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Placement Readiness",
    layout="wide"
)

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    os.makedirs("model", exist_ok=True)

    if not os.path.exists("model/readiness_model.pkl"):
        train_and_save_model()

    model = joblib.load("model/readiness_model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    scaler = joblib.load("model/scaler.pkl")

    return model, encoder, scaler

model, encoder, scaler = load_artifacts()

# ---------------- HEADER ----------------
st.title("🎓 Student Placement Readiness Dashboard")
st.caption("ML-powered assessment of placement readiness with visual insights")

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 Student Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
degree = st.sidebar.selectbox("Degree", ["B.Tech", "M.Tech", "B.Sc", "M.Sc"])
branch = st.sidebar.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)

coding = st.sidebar.slider("Coding Skills (1–10)", 1, 10, 6)
communication = st.sidebar.slider("Communication Skills (1–10)", 1, 10, 6)

projects = st.sidebar.number_input("Projects", 0, 10, 2)
internships = st.sidebar.number_input("Internships", 0, 5, 1)

if st.sidebar.button("🚀 Evaluate Readiness"):
    st.session_state.evaluated = True

# ---------------- MAIN OUTPUT ----------------
if st.session_state.evaluated:

    # ---------- INPUT ----------
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

    X_cat = encoder.transform(input_df[cat_cols])
    X_num = scaler.transform(input_df[num_cols])

    X = np.hstack([X_num, X_cat])

    predicted_score = float(model.predict(X)[0])
    level = readiness_level(predicted_score)

    # ================= CONTAINER 1 =================
    with st.container():
        st.subheader("📊 Readiness Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{predicted_score:.1f} / 100")
        c2.metric("Level", level)

        if level == "Low":
            c3.error("Needs Improvement")
        elif level == "Medium":
            c3.warning("Moderately Prepared")
        else:
            c3.success("Well Prepared")

    st.divider()

    # ================= CONTAINER 2 =================
    with st.container():
        st.subheader("📈 Skill Radar")

        radar_labels = ["CGPA", "Coding", "Communication", "Projects", "Internships"]
        radar_values = [
            cgpa,
            coding,
            communication,
            min(projects, 5) * 2,
            min(internships, 5) * 2
        ]

        fig = go.Figure(go.Scatterpolar(
            r=radar_values,
            theta=radar_labels,
            fill="toself"
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 10])),
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ================= CONTAINER 3 =================
    with st.container():
        st.subheader("🧩 Recommendations")

        if cgpa < 7:
            st.write("📘 Improve CGPA through consistent study.")
        if coding < 7:
            st.write("💻 Practice DSA and coding daily.")
        if projects < 3:
            st.write("🛠️ Build more real-world projects.")
        if internships < 1:
            st.write("🏢 Apply for internships early.")

else:
    st.info("👈 Fill the form and click **Evaluate Readiness**")

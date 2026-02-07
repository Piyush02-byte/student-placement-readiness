🎓 PlacePrep Pro — Student Placement Readiness Intelligence (v1.0)
        
Project Status: ✅ Phase 1 Complete (v1.0 – Stable)

PlacePrep Pro is an end-to-end machine learning–powered web application that evaluates a student’s placement readiness based on academics, technical skills, soft skills, and real-world experience.
It delivers a quantitative readiness score (0–100), rich visual analytics, and actionable, personalized recommendations to help students understand where they stand and what to improve next.

🌐 Live Application:
    👉 https://student-placement-readiness.streamlit.app/

Phase-1 Focus:
Building a robust, explainable, and user-centric placement readiness system — not just a prediction model.

Roadmap
•	v1.0 (Current): Placement readiness prediction with analytics dashboard
•	v2.0 (Planned): Company-specific readiness, profile history, PDF reports


🚀 Why PlacePrep Pro?
Most students preparing for campus placements lack:
•	A clear benchmark of their readiness
•	Data-driven feedback (beyond vague advice)
•	Prioritized action steps

Campus placement preparation is often fragmented across CGPA, skills, and experience. 
       PlacePrep Pro unifies these factors into a single, interpretable readiness score with visual insights and actionable guidance.

✨ Key Features

📊 Placement Readiness Score
•	Predicts an overall readiness score on a 0–100 scale
•	Categorizes students into Low, Medium, High readiness
•	Backed by a trained ML regression model (not hard-coded logic)

📈 Rich Visual Analytics
•	Radar chart: Skill balance vs ideal profile
•	Contribution bar chart: What actually drives your score
•	Component score bars: Individual strength assessment
•	Gauge meter: Overall placement preparedness at a glance

🎯 Personalized Action Plan
•	Automatically identifies weakest and strongest areas
•	Generates prioritized recommendations:
•	Critical (must fix)
•	Important (next focus)
•	Optional (nice-to-have)
•	Includes timelines and impact context

🎨 Premium UX/UI
•	Clean, modern layout with custom CSS
•	Branded sidebar and hero section
•	Smooth analysis simulation for better user experience
•	Responsive, presentation-ready dashboard

🧠 Machine Learning Approach
Model
•	Algorithm: Random Forest Regressor
•	Target: Placement Readiness Score (0–100)
•	Why Regression?
•	Avoids rigid classification
•	Produces smooth, realistic score variations
•	Better reflects real-world readiness

      Feature Set

Numerical
CGPA
Coding Skills
Communication Skills
Number of Projects
Number of Internships

Categorical
Gender
Degree
Branch

Preprocessing
One-Hot Encoding for categorical features
Standard Scaling for numerical features
Encapsulated in a reusable StudentDataPreprocessor class

Model Outputs
Continuous readiness score
Feature importance analysis
Performance metrics (MAE, RMSE, R²)


🏗️ Project Structure

       student-placement-readiness/
│
├── app.py                     # Streamlit application (UI + inference)
├── train_model.py             # Model training & evaluation pipeline
├── preprocessing.py           # Encoding, scaling & feature engineering
├── scoring.py                 # Readiness level logic
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
│
├── data/
│   └── raw_students.csv       # Training dataset
│
├── model/
│   ├── readiness_model.pkl    # Trained ML model
│   ├── encoder.pkl            # OneHotEncoder
│   ├── scaler.pkl             # StandardScaler
│   ├── metrics.pkl            # Model evaluation metrics
│   └── feature_importance.csv # Feature importance



         🖥️ Running Locally

1️⃣ Clone the Repository
git clone https://github.com/Piyush02-byte/student-placement-readiness.git
cd student-placement-readiness

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (first time only)
python train_model.py

4️⃣ Run the App
streamlit run app.py

The app will open at:
👉 http://localhost:8501


   📊 Example Use Cases
•	🎓 Students – Self-assess placement readiness and plan improvements.
•	🧑‍🏫 Career counselors – Provide data-driven guidance.
•	🏫 Institutions – Analyze overall student preparedness.
•	💼 Recruitment prep – Understand skill gaps before interviews.


📌 Phase-1 Scope (v1.0)
✔ End-to-end ML pipeline
✔ Clean modular architecture
✔ Production-deployed web app
✔ Multiple analytical charts
✔ Personalized recommendations

	Phase-1 focuses on individual assessment.
Future phases may introduce tracking, comparisons, and advanced analytics.


🔮 Planned Enhancements (Phase-2 Ideas)
•	User accounts & history tracking
•	PDF/Excel readiness reports
•	Company-specific readiness scoring
•	Interview question recommendations
•	Progress tracking over time
(Not implemented yet – intentionally out of scope for v1.0)


🧑‍💻 Author
Piyush Kumar
B.Tech (Computer Science & Engineering), 3rd Year
Central University of Haryana
Diploma in Electronics Engineering, Government Polytechnic Muzaffarpur
        Interests: Machine Learning, Data Science, and applied software systems
🔗 GitHub: https://github.com/Piyush02-byte



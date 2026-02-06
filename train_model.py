import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


def train_and_save_model():
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv("data/raw_students.csv")

    cat_cols = ["Gender", "Degree", "Branch"]
    num_cols = ["CGPA", "Internships", "Projects", "Coding_Skills", "Communication_Skills"]

    # ---------------- CREATE TARGET ----------------
    df["Readiness_Score"] = (
        df["CGPA"] * 10 +
        df["Coding_Skills"] * 6 +
        df["Communication_Skills"] * 4 +
        df["Projects"] * 5 +
        df["Internships"] * 8
    ).clip(0, 100)

    X_cat = df[cat_cols]
    X_num = df[num_cols]
    y = df["Readiness_Score"]

    # ---------------- PREPROCESSING ----------------
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_enc = encoder.fit_transform(X_cat)

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    X = np.hstack([X_num_scaled, X_cat_enc])

    # ---------------- MODEL ----------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    # ---------------- SAVE ARTIFACTS ----------------
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/readiness_model.pkl")
    joblib.dump(encoder, "model/encoder.pkl")
    joblib.dump(scaler, "model/scaler.pkl")


if __name__ == "__main__":
    train_and_save_model()

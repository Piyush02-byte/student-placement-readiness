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

    # ---------------- TARGET (CONTROLLED) ----------------
    df["Readiness_Score"] = (
        df["CGPA"] * 10 +
        df["Coding_Skills"] * 6 +
        df["Communication_Skills"] * 4 +
        df["Projects"] * 5 +
        df["Internships"] * 8
    ).clip(0, 100)

    y = df["Readiness_Score"]

    # ---------------- PREPROCESSING ----------------
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(df[cat_cols])

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    X = np.hstack([X_num, X_cat])

    # ---------------- MODEL ----------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    # ---------------- SAVE ----------------
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/readiness_model.pkl")
    joblib.dump(encoder, "model/encoder.pkl")
    joblib.dump(scaler, "model/scaler.pkl")


if __name__ == "__main__":
    train_and_save_model()

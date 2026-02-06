import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from scoring import calculate_readiness
from preprocessing import Preprocessor

# Load dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_students.csv")

df = pd.read_csv(DATA_PATH)


# Drop unused columns
df = df.drop(columns=["Student_ID", "Age"])

# Create target
df["Readiness_Score"] = df.apply(calculate_readiness, axis=1)

X = df.drop(columns=["Readiness_Score"])
y = df["Readiness_Score"]

# Preprocessing
preprocessor = Preprocessor()
preprocessor.fit(X)
X_processed = preprocessor.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "model/readiness_model.pkl")
joblib.dump(preprocessor.encoder, "model/encoder.pkl")
joblib.dump(preprocessor.scaler, "model/scaler.pkl")

print("✅ Model and preprocessors saved successfully")

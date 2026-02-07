"""
Student Placement Readiness Model Training Module
Trains a Random Forest model to predict placement readiness scores
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_readiness_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate readiness score based on weighted features
    
    Args:
        df: DataFrame containing student features
        
    Returns:
        Series containing readiness scores (0-100)
    """
    score = (
        df["CGPA"] * 10 +           # Max contribution: 100
        df["Coding_Skills"] * 6 +    # Max contribution: 60
        df["Communication_Skills"] * 4 +  # Max contribution: 40
        df["Projects"] * 5 +         # Max contribution: 50
        df["Internships"] * 8        # Max contribution: 40
    )
    
    # Normalize to 0-100 scale
    return score.clip(0, 100)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate input data for required columns and value ranges
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    required_cols = ["Gender", "Degree", "Branch", "CGPA", "Internships", 
                     "Projects", "Coding_Skills", "Communication_Skills"]
    
    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values
    if df[required_cols].isnull().any().any():
        raise ValueError("Dataset contains null values")
    
    # Validate value ranges
    if not df["CGPA"].between(0, 10).all():
        raise ValueError("CGPA must be between 0 and 10")
    
    if not df["Coding_Skills"].between(1, 10).all():
        raise ValueError("Coding_Skills must be between 1 and 10")
    
    if not df["Communication_Skills"].between(1, 10).all():
        raise ValueError("Communication_Skills must be between 1 and 10")
    
    logger.info("Data validation passed")
    return True


def train_and_save_model(data_path: str = "data/raw_students.csv", 
                         model_dir: str = "model",
                         test_size: float = 0.2) -> dict:
    """
    Train Random Forest model and save artifacts
    
    Args:
        data_path: Path to training data CSV
        model_dir: Directory to save model artifacts
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary containing training metrics
    """
    logger.info("Starting model training pipeline...")
    
    # ---------------- LOAD DATA ----------------
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from {data_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # ---------------- VALIDATE DATA ----------------
    validate_data(df)
    
    # Define column types
    cat_cols = ["Gender", "Degree", "Branch"]
    num_cols = ["CGPA", "Internships", "Projects", "Coding_Skills", "Communication_Skills"]
    
    # ---------------- CREATE TARGET ----------------
    df["Readiness_Score"] = calculate_readiness_score(df)
    logger.info(f"Readiness score range: {df['Readiness_Score'].min():.2f} - {df['Readiness_Score'].max():.2f}")
    
    y = df["Readiness_Score"]
    
    # ---------------- PREPROCESSING ----------------
    logger.info("Preprocessing features...")
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(df[cat_cols])
    logger.info(f"Encoded {len(cat_cols)} categorical features into {X_cat.shape[1]} features")
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])
    
    # Combine features
    X = np.hstack([X_num, X_cat])
    logger.info(f"Final feature matrix shape: {X.shape}")
    
    # ---------------- TRAIN/TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # ---------------- MODEL TRAINING ----------------
    logger.info("Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # ---------------- EVALUATION ----------------
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    metrics = {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "n_samples": len(df),
        "n_features": X.shape[1]
    }
    
    logger.info("=" * 50)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("=" * 50)
    logger.info(f"Training MAE:   {train_mae:.2f}")
    logger.info(f"Testing MAE:    {test_mae:.2f}")
    logger.info(f"Training R²:    {train_r2:.4f}")
    logger.info(f"Testing R²:     {test_r2:.4f}")
    logger.info("=" * 50)
    
    # ---------------- SAVE ARTIFACTS ----------------
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = os.path.join(model_dir, "readiness_model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    metrics_path = os.path.join(model_dir, "metrics.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(metrics, metrics_path)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Encoder saved to: {encoder_path}")
    logger.info(f"Scaler saved to: {scaler_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = train_and_save_model()
        logger.info("✅ Model training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        raise
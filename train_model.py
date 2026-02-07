"""
Fixed training module with proper score normalization
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from preprocessing import StudentDataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_readiness_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate readiness score with PROPER NORMALIZATION
    
    The original formula was creating scores that exceeded 100, causing
    the model to only predict edge values (0, 50, 100)
    
    Args:
        df: DataFrame with student features
        
    Returns:
        Normalized readiness scores (0-100)
    """
    # Define maximum possible contributions
    MAX_CGPA_CONTRIB = 10 * 10  # 100
    MAX_CODING_CONTRIB = 10 * 6  # 60
    MAX_COMM_CONTRIB = 10 * 4  # 40
    MAX_PROJ_CONTRIB = 10 * 5  # 50 (assuming max 10 projects)
    MAX_INTERN_CONTRIB = 5 * 8  # 40 (assuming max 5 internships)
    
    # Total maximum possible score
    MAX_SCORE = (MAX_CGPA_CONTRIB + MAX_CODING_CONTRIB + 
                 MAX_COMM_CONTRIB + MAX_PROJ_CONTRIB + MAX_INTERN_CONTRIB)
    # MAX_SCORE = 290
    
    # Calculate raw score
    raw_score = (
        df["CGPA"] * 10 +
        df["Coding_Skills"] * 6 +
        df["Communication_Skills"] * 4 +
        df["Projects"].clip(0, 10) * 5 +  # Cap projects at 10
        df["Internships"].clip(0, 5) * 8   # Cap internships at 5
    )
    
    # Normalize to 0-100 scale
    normalized_score = (raw_score / MAX_SCORE) * 100
    
    # Ensure within bounds
    normalized_score = normalized_score.clip(0, 100)
    
    logger.info(f"Score stats - Min: {normalized_score.min():.2f}, "
                f"Max: {normalized_score.max():.2f}, "
                f"Mean: {normalized_score.mean():.2f}, "
                f"Std: {normalized_score.std():.2f}")
    
    return normalized_score


def train_and_save_model(data_path: str = "data/raw_students.csv", 
                         model_dir: str = "model",
                         test_size: float = 0.2,
                         random_state: int = 42) -> dict:
    """
    Train placement readiness model with proper preprocessing
    
    Args:
        data_path: Path to training data
        model_dir: Directory to save artifacts
        test_size: Test set fraction
        random_state: Random seed
        
    Returns:
        Dictionary of training metrics
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # ============ LOAD DATA ============
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✓ Loaded {len(df)} records from {data_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # ============ CREATE TARGET ============
    df["Readiness_Score"] = calculate_readiness_score(df)
    
    # Check for score distribution issues
    unique_scores = df["Readiness_Score"].nunique()
    logger.info(f"Unique target values: {unique_scores}")
    
    if unique_scores < 10:
        logger.warning("⚠️  Very few unique target values - check score calculation!")
    
    # ============ PREPROCESSING ============
    preprocessor = StudentDataPreprocessor(model_dir=model_dir)
    
    # Separate features and target
    feature_cols = (preprocessor.categorical_cols + 
                   preprocessor.numerical_cols)
    
    X = df[feature_cols]
    y = df["Readiness_Score"]
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    logger.info(f"✓ Feature matrix shape: {X_transformed.shape}")
    logger.info(f"✓ Feature names: {preprocessor.get_feature_names()}")
    
    # ============ TRAIN/TEST SPLIT ============
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"✓ Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ============ MODEL TRAINING ============
    logger.info("Training Random Forest Regressor...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,  # Increased depth
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Better generalization
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # ============ EVALUATION ============
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Clip predictions to valid range
    train_pred = np.clip(train_pred, 0, 100)
    test_pred = np.clip(test_pred, 0, 100)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Check prediction distribution
    logger.info(f"Train predictions - Min: {train_pred.min():.2f}, "
                f"Max: {train_pred.max():.2f}, "
                f"Mean: {train_pred.mean():.2f}")
    logger.info(f"Test predictions - Min: {test_pred.min():.2f}, "
                f"Max: {test_pred.max():.2f}, "
                f"Mean: {test_pred.mean():.2f}")
    
    metrics = {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "n_samples": len(df),
        "n_features": X_transformed.shape[1],
        "unique_predictions": len(np.unique(test_pred))
    }
    
    # Display metrics
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("=" * 60)
    logger.info(f"Training MAE:      {train_mae:.3f}")
    logger.info(f"Testing MAE:       {test_mae:.3f}")
    logger.info(f"Training RMSE:     {train_rmse:.3f}")
    logger.info(f"Testing RMSE:      {test_rmse:.3f}")
    logger.info(f"Training R²:       {train_r2:.4f}")
    logger.info(f"Testing R²:        {test_r2:.4f}")
    logger.info(f"Unique predictions: {metrics['unique_predictions']}")
    logger.info("=" * 60)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': preprocessor.get_feature_names(),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Important Features:")
    logger.info(feature_importance.head(10).to_string(index=False))
    
    # ============ SAVE ARTIFACTS ============
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = Path(model_dir) / "readiness_model.pkl"
    metrics_path = Path(model_dir) / "metrics.pkl"
    importance_path = Path(model_dir) / "feature_importance.csv"
    
    joblib.dump(model, model_path)
    joblib.dump(metrics, metrics_path)
    feature_importance.to_csv(importance_path, index=False)
    
    # Save preprocessor
    preprocessor.save(model_dir)
    
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info(f"✓ Metrics saved to: {metrics_path}")
    logger.info(f"✓ Feature importance saved to: {importance_path}")
    
    logger.info("=" * 60)
    logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = train_and_save_model()
        
        # Validation check
        if metrics['unique_predictions'] < 20:
            logger.error("❌ Model is producing too few unique predictions!")
            logger.error("This suggests an issue with the training data or score calculation.")
        else:
            logger.info("✅ Model validation passed!")
            
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise
"""
Preprocessing module for Student Placement Readiness System
Handles encoding, scaling, and feature engineering
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Optional, Tuple

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessor for student placement data
    Handles categorical encoding, numerical scaling, and feature engineering
    """
    
    def __init__(self, model_dir: str = "model"):
        """
        Initialize preprocessor
        
        Args:
            model_dir: Directory to save/load preprocessing artifacts
        """
        self.encoder = OneHotEncoder(
            sparse_output=False, 
            handle_unknown="ignore",
            drop=None  # Keep all categories for transparency
        )
        # Using StandardScaler instead of MinMaxScaler for better generalization
        self.scaler = StandardScaler()
        
        self.categorical_cols = ["Gender", "Degree", "Branch"]
        self.numerical_cols = [
            "CGPA",
            "Internships",
            "Projects",
            "Coding_Skills",
            "Communication_Skills"
        ]
        
        self.model_dir = Path(model_dir)
        self.is_fitted = False
        
        # Store feature names after encoding
        self.feature_names_ = None
        self.n_features_in_ = None
        
    def validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data structure and values
        
        Args:
            X: Input DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        required_cols = self.categorical_cols + self.numerical_cols
        
        # Check columns exist
        missing = set(required_cols) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check for nulls
        null_cols = X[required_cols].columns[X[required_cols].isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"Null values found in columns: {null_cols}")
        
        # Validate numerical ranges
        if not X["CGPA"].between(0, 10).all():
            logger.warning(f"CGPA out of range [0-10]: {X['CGPA'].describe()}")
            
        if not X["Coding_Skills"].between(1, 10).all():
            logger.warning(f"Coding_Skills out of range [1-10]: {X['Coding_Skills'].describe()}")
            
        if not X["Communication_Skills"].between(1, 10).all():
            logger.warning(f"Communication_Skills out of range [1-10]: {X['Communication_Skills'].describe()}")
            
        logger.info("✓ Input validation passed")
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit encoder and scaler on training data
        
        Args:
            X: Training DataFrame
            y: Not used (for sklearn compatibility)
            
        Returns:
            self
        """
        logger.info("Fitting preprocessor...")
        
        # Validate
        self.validate_input(X)
        
        # Fit transformers
        self.encoder.fit(X[self.categorical_cols])
        self.scaler.fit(X[self.numerical_cols])
        
        # Store metadata
        self.n_features_in_ = len(self.categorical_cols) + len(self.numerical_cols)
        
        # Get feature names after encoding
        cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
        self.feature_names_ = list(self.numerical_cols) + list(cat_feature_names)
        
        self.is_fitted = True
        
        logger.info(f"✓ Fitted on {len(X)} samples")
        logger.info(f"✓ Total features after encoding: {len(self.feature_names_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input data using fitted encoder and scaler
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Validate
        self.validate_input(X)
        
        # Transform categorical features
        encoded_cat = self.encoder.transform(X[self.categorical_cols])
        
        # Transform numerical features
        scaled_num = self.scaler.transform(X[self.numerical_cols])
        
        # Combine features (numerical first, then categorical)
        X_transformed = np.hstack([scaled_num, encoded_cat])
        
        logger.debug(f"Transformed shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X: Input DataFrame
            y: Not used
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform_numerical(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform numerical features only
        
        Args:
            X_scaled: Scaled numerical features
            
        Returns:
            Original scale numerical features
        """
        return self.scaler.inverse_transform(X_scaled)
    
    def get_feature_names(self) -> list:
        """Get feature names after transformation"""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.feature_names_
    
    def save(self, model_dir: Optional[str] = None) -> None:
        """
        Save preprocessor artifacts
        
        Args:
            model_dir: Directory to save artifacts (uses self.model_dir if None)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        save_dir = Path(model_dir) if model_dir else self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        encoder_path = save_dir / "encoder.pkl"
        scaler_path = save_dir / "scaler.pkl"
        metadata_path = save_dir / "preprocessor_metadata.pkl"
        
        joblib.dump(self.encoder, encoder_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'feature_names': self.feature_names_,
            'n_features_in': self.n_features_in_
        }
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"✓ Saved encoder to: {encoder_path}")
        logger.info(f"✓ Saved scaler to: {scaler_path}")
        logger.info(f"✓ Saved metadata to: {metadata_path}")
    
    @classmethod
    def load(cls, model_dir: str = "model") -> 'StudentDataPreprocessor':
        """
        Load preprocessor from saved artifacts
        
        Args:
            model_dir: Directory containing saved artifacts
            
        Returns:
            Loaded preprocessor instance
        """
        load_dir = Path(model_dir)
        
        encoder_path = load_dir / "encoder.pkl"
        scaler_path = load_dir / "scaler.pkl"
        metadata_path = load_dir / "preprocessor_metadata.pkl"
        
        # Check files exist
        if not all([p.exists() for p in [encoder_path, scaler_path]]):
            raise FileNotFoundError(f"Preprocessor artifacts not found in {model_dir}")
        
        # Create instance and load
        preprocessor = cls(model_dir=str(load_dir))
        preprocessor.encoder = joblib.load(encoder_path)
        preprocessor.scaler = joblib.load(scaler_path)
        
        # Load metadata if exists
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            preprocessor.categorical_cols = metadata['categorical_cols']
            preprocessor.numerical_cols = metadata['numerical_cols']
            preprocessor.feature_names_ = metadata['feature_names']
            preprocessor.n_features_in_ = metadata['n_features_in']
        
        preprocessor.is_fitted = True
        
        logger.info(f"✓ Loaded preprocessor from: {model_dir}")
        
        return preprocessor
    
    def get_stats(self) -> dict:
        """Get preprocessing statistics"""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        
        return {
            'n_categorical_features': len(self.categorical_cols),
            'n_numerical_features': len(self.numerical_cols),
            'n_encoded_features': len(self.feature_names_),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist(),
            'categorical_categories': {
                col: list(cats) 
                for col, cats in zip(
                    self.categorical_cols, 
                    self.encoder.categories_
                )
            }
        }


# Backward compatibility function
def create_legacy_preprocessor():
    """Create preprocessor with legacy interface"""
    return StudentDataPreprocessor()
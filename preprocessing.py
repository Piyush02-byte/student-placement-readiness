import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class Preprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = MinMaxScaler()

        self.categorical_cols = ["Gender", "Degree", "Branch"]
        self.numerical_cols = [
            "CGPA",
            "Internships",
            "Projects",
            "Coding_Skills",
            "Communication_Skills"
        ]

    def fit(self, X):
        self.encoder.fit(X[self.categorical_cols])
        self.scaler.fit(X[self.numerical_cols])

    def transform(self, X):
        encoded_cat = self.encoder.transform(X[self.categorical_cols])
        scaled_num = self.scaler.transform(X[self.numerical_cols])

        X_final = np.hstack([scaled_num, encoded_cat])
        return X_final

    def save(self):
        joblib.dump(self.encoder, "encoder.pkl")
        joblib.dump(self.scaler, "scaler.pkl")

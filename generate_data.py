"""
Generate realistic training data with variance
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_training_data(n_samples=1000, output_path="data/raw_students.csv"):
    """
    Generate realistic student data with natural variance
    """
    print(f"Generating {n_samples} training samples...")
    
    # Generate diverse data
    data = {
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "Degree": np.random.choice(["B.Tech", "M.Tech", "B.Sc", "M.Sc"], n_samples, 
                                   p=[0.5, 0.2, 0.2, 0.1]),
        "Branch": np.random.choice(["CSE", "ECE", "ME", "CE", "EE"], n_samples,
                                   p=[0.35, 0.25, 0.15, 0.15, 0.10])
    }
    
    # CGPA: Normal distribution around 7.5
    data["CGPA"] = np.clip(np.random.normal(7.5, 1.2, n_samples), 5.0, 10.0)
    
    # Coding Skills: Slightly correlated with branch
    base_coding = np.random.randint(3, 9, n_samples)
    cse_boost = np.where(data["Branch"] == "CSE", 
                         np.random.randint(0, 3, n_samples), 0)
    data["Coding_Skills"] = np.clip(base_coding + cse_boost, 1, 10)
    
    # Communication: Independent random
    data["Communication_Skills"] = np.random.randint(4, 10, n_samples)
    
    # Projects: Slightly correlated with coding
    project_base = (data["Coding_Skills"] / 2).astype(int)
    project_noise = np.random.randint(-2, 3, n_samples)
    data["Projects"] = np.clip(project_base + project_noise, 0, 10)
    
    # Internships: Correlated with CGPA
    intern_base = ((data["CGPA"] - 5) / 2).astype(int)
    intern_noise = np.random.randint(-1, 2, n_samples)
    data["Internships"] = np.clip(intern_base + intern_noise, 0, 5)
    
    df = pd.DataFrame(data)
    
    # Calculate readiness score with REALISTIC WEIGHTS + NOISE
    # This prevents the model from learning a perfect deterministic function
    
    # Normalize each component to 0-1 scale first
    cgpa_norm = (df["CGPA"] - 5) / 5  # 5-10 -> 0-1
    coding_norm = (df["Coding_Skills"] - 1) / 9  # 1-10 -> 0-1
    comm_norm = (df["Communication_Skills"] - 1) / 9  # 1-10 -> 0-1
    proj_norm = df["Projects"] / 10  # 0-10 -> 0-1
    intern_norm = df["Internships"] / 5  # 0-5 -> 0-1
    
    # Weighted average (weights sum to 1)
    weights = {
        'cgpa': 0.30,      # 30% weight
        'coding': 0.25,    # 25% weight
        'comm': 0.15,      # 15% weight
        'projects': 0.20,  # 20% weight
        'internships': 0.10 # 10% weight
    }
    
    base_score = (
        cgpa_norm * weights['cgpa'] +
        coding_norm * weights['coding'] +
        comm_norm * weights['comm'] +
        proj_norm * weights['projects'] +
        intern_norm * weights['internships']
    ) * 100  # Scale to 0-100
    
    # Add realistic noise (±5 points)
    noise = np.random.normal(0, 3, n_samples)
    df["Readiness_Score"] = np.clip(base_score + noise, 0, 100)
    
    # Statistics
    print("\n" + "="*60)
    print("GENERATED DATA STATISTICS")
    print("="*60)
    print(f"\nReadiness Score Distribution:")
    print(df["Readiness_Score"].describe())
    print(f"\nScore Range: {df['Readiness_Score'].min():.2f} - {df['Readiness_Score'].max():.2f}")
    print(f"Unique Scores: {df['Readiness_Score'].nunique()}")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {n_samples} samples to {output_path}")
    print("="*60)
    
    return df


if __name__ == "__main__":
    df = generate_training_data(n_samples=1000)
    
    # Show sample
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    # Show distribution by level
    def readiness_level(score):
        if score < 60:
            return "Low"
        elif score < 80:
            return "Medium"
        else:
            return "High"
    
    df['Level'] = df['Readiness_Score'].apply(readiness_level)
    print("\nDistribution by Level:")
    print(df['Level'].value_counts())

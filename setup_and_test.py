"""
Complete setup and testing script
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and show output"""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print('='*70)
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Completed: {description}")

def main():
    print("\n" + "="*70)
    print("STUDENT PLACEMENT READINESS - COMPLETE SETUP")
    print("="*70)
    
    # Step 1: Generate data
    run_command(
        "python generate_data.py",
        "Generating training data"
    )
    
    # Step 2: Train model
    run_command(
        "python train_model.py",
        "Training model"
    )
    
    # Step 3: Verify outputs
    print("\n" + "="*70)
    print("VERIFYING OUTPUTS")
    print("="*70)
    
    required_files = [
        "data/raw_students.csv",
        "model/readiness_model.pkl",
        "model/encoder.pkl",
        "model/scaler.pkl"
    ]
    
    all_good = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            all_good = False
    
    if all_good:
        print("\n" + "="*70)
        print("✅ SETUP COMPLETE!")
        print("="*70)
        print("\nRun the Streamlit app:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Setup incomplete - check errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()
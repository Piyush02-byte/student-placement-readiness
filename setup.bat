@echo off
echo ====================================
echo Student Placement Readiness Setup
echo ====================================

echo.
echo [1/5] Creating virtual environment...
python -m venv venv

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/5] Installing dependencies...
pip install pandas numpy scikit-learn joblib streamlit plotly

echo.
echo [5/5] Verifying installation...
python -c "import pandas, numpy, sklearn, joblib, streamlit, plotly; print('✓ All packages installed successfully!')"

echo.
echo ====================================
echo ✓ Setup Complete!
echo ====================================
echo.
echo Next steps:
echo   1. Run: venv\Scripts\activate
echo   2. Run: python generate_data.py
echo   3. Run: python train_model.py
echo   4. Run: streamlit run app.py
echo.
pause
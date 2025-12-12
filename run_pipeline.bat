@echo off
REM Full pipeline script for Shodh AI Loan Project
REM Run this from the project root directory

echo ============================================================
echo Shodh AI Loan Project - Full Pipeline
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo Step 1: Download and Sample Data
echo ============================================================
cd src
python 00_download_and_sample.py --input ../data/accepted_2007_to_2018.csv --sample-size 200000

echo.
echo ============================================================
echo Step 2: Preprocess Data
echo ============================================================
python 01_preprocess.py

echo.
echo ============================================================
echo Step 3: Train Supervised Model
echo ============================================================
python 03_train_supervised.py --epochs 30 --batch-size 512

echo.
echo ============================================================
echo Step 4: Evaluate Supervised Model
echo ============================================================
python 04_eval_supervised.py

echo.
echo ============================================================
echo Step 5: Convert Data for RL
echo ============================================================
python 05_convert_for_rl.py

echo.
echo ============================================================
echo Step 6: Train Offline RL Agent
echo ============================================================
python 06_train_offline_rl.py --n-epochs 50

echo.
echo ============================================================
echo Step 7: Evaluate and Compare RL vs Supervised
echo ============================================================
python 07_eval_rl.py

cd ..

echo.
echo ============================================================
echo Pipeline Complete!
echo ============================================================
echo.
echo Results are in:
echo   - models/best_model.pth (supervised model)
echo   - models/rl/cql_model.d3 (RL agent)
echo   - models/test_results.json (supervised metrics)
echo   - models/rl/rl_evaluation.json (RL metrics)
echo.
echo Next steps:
echo   1. Review the generated plots in models/
echo   2. Check policy_comparisons.csv for decision differences
echo   3. Fill in reports/REPORT_TEMPLATE.md with your results
echo   4. Create final PDF report
echo.
pause

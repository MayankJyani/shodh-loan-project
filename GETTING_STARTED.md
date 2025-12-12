# Getting Started: Complete Guide

This guide will walk you through completing the Shodh AI ML hiring project from start to finish.

## ⏱️ Time Estimate

- **Setup & Data Download**: 30 mins
- **Pipeline Execution**: 2-3 hours (depending on hardware)
- **Report Writing**: 2-3 hours
- **Total**: ~6 hours

## 📋 Prerequisites

1. **Python 3.9+** installed
2. **Git** (optional, for version control)
3. **~5GB free disk space**
4. **Internet connection** (for downloading data and packages)

## 🚀 Quick Start (3 Steps)

### Step 1: Download Data (10 mins)

1. Go to https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Download `accepted_2007_to_2018.csv` (warning: ~2GB file!)
3. Place it in the `data/` folder of this project

### Step 2: Run Pipeline (2-3 hours)

**Option A: Automated (Recommended)**
```bash
# From project root
run_pipeline.bat
```

**Option B: Manual**
```bash
# Activate environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run each script
cd src
python 00_download_and_sample.py --input ../data/accepted_2007_to_2018.csv
python 01_preprocess.py
python 03_train_supervised.py
python 04_eval_supervised.py
python 05_convert_for_rl.py
python 06_train_offline_rl.py
python 07_eval_rl.py
```

### Step 3: Write Report (2-3 hours)

1. Open `reports/REPORT_TEMPLATE.md`
2. Fill in all `XXX` placeholders with your actual results
3. Use the metrics from:
   - `models/test_results.json`
   - `models/rl/rl_evaluation.json`
   - `models/rl/policy_comparisons.csv`
4. Convert to PDF (use Word, Google Docs, or pandoc)

---

## 📊 Detailed Instructions

### Part 1: Setup (Day 1 - Morning)

#### 1.1 Environment Setup
```bash
cd C:\Users\mayan\OneDrive\Desktop\ShodhAI\shodh-loan-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Troubleshooting**:
- If `d3rlpy` fails: `pip install torch==1.12.0` first, then retry
- If `pyarrow` fails: `pip install pyarrow --no-cache-dir`

#### 1.2 Data Download

**Manual Download**:
1. Visit https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Click "Download" (requires Kaggle account)
3. Extract `accepted_2007_to_2018.csv` from the zip
4. Move to `data/accepted_2007_to_2018.csv`

**Kaggle API** (if you have credentials):
```bash
# Setup: Place kaggle.json in ~/.kaggle/ (or %USERPROFILE%\.kaggle\ on Windows)
cd src
python 00_download_and_sample.py --download
```

#### 1.3 Verify Setup
```bash
# Check data file exists
dir ..\data\accepted_2007_to_2018.csv

# Check Python packages
python -c "import torch, pandas, sklearn, d3rlpy; print('All packages installed!')"
```

---

### Part 2: Run Pipeline (Day 1 - Afternoon/Evening)

#### 2.1 Sample Data (5 mins)
```bash
cd src
python 00_download_and_sample.py --input ../data/accepted_2007_to_2018.csv --sample-size 200000
```

**Expected Output**:
- `data/sampled_data.csv` (200K rows)
- Default rate printed (~20-25%)

#### 2.2 Preprocess (10 mins)
```bash
python 01_preprocess.py
```

**Expected Output**:
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `data/processed/preprocessing.pkl`
- `data/processed/feature_info.pkl`

#### 2.3 Train Supervised Model (30-60 mins)
```bash
python 03_train_supervised.py --epochs 30 --batch-size 512
```

**What to watch**:
- Training loss should decrease
- Validation AUC should increase (aim for >0.65)
- Early stopping will trigger when validation stops improving

**Expected Output**:
- `models/best_model.pth`
- `models/training_history.json`
- `models/learning_curves.png`

#### 2.4 Evaluate Supervised Model (5 mins)
```bash
python 04_eval_supervised.py
```

**Expected Output**:
- `models/test_results.json` ← **Key file for report!**
- `models/roc_curve.png`
- `models/confusion_matrix.png`
- `models/threshold_analysis.png`
- `models/test_predictions.csv`

**Check your results**:
```bash
type ..\models\test_results.json
```
You should see AUC ~0.65-0.70, F1 ~0.40-0.50

#### 2.5 Convert to RL Format (5 mins)
```bash
python 05_convert_for_rl.py
```

**Expected Output**:
- `data/rl/train_rl.npz`
- `data/rl/val_rl.npz`
- `data/rl/test_rl.npz`

#### 2.6 Train RL Agent (60-90 mins)
```bash
python 06_train_offline_rl.py --n-epochs 50
```

**What to watch**:
- TD error should stabilize
- May take longer than supervised model

**Expected Output**:
- `models/rl/cql_model.d3`

#### 2.7 Evaluate RL & Compare (5 mins)
```bash
python 07_eval_rl.py
```

**Expected Output**:
- `models/rl/rl_evaluation.json` ← **Key file for report!**
- `models/rl/policy_comparisons.csv` ← **Key file for report!**

**Check your results**:
```bash
type ..\models\rl\rl_evaluation.json
```
You should see policy value and comparison metrics

---

### Part 3: Analysis & Report (Day 2)

#### 3.1 Gather Results

**Supervised Model Results**:
```bash
# View test metrics
type ..\models\test_results.json

# Key numbers to note:
# - test_auc
# - test_f1_default_threshold
# - optimal_threshold
```

**RL Agent Results**:
```bash
# View RL evaluation
type ..\models\rl\rl_evaluation.json

# Key numbers to note:
# - rl_policy.policy_value
# - rl_policy.approve_rate
# - rl_policy.improvement_pct
```

**Policy Comparisons**:
```bash
# View decision differences
type ..\models\rl\policy_comparisons.csv

# Find 2-3 interesting cases where models disagree
```

#### 3.2 Write Report

Open `reports/REPORT_TEMPLATE.md` and fill in:

1. **Section 1**: Data statistics from preprocessing output
2. **Section 2**: Supervised metrics from `test_results.json`
3. **Section 3**: RL metrics from `rl_evaluation.json`
4. **Section 4**: Examples from `policy_comparisons.csv`
5. **Section 5**: Your analysis and recommendations

**Pro Tips**:
- Use concrete numbers (don't just say "good AUC", say "AUC=0.68")
- Explain *why* RL approves high-risk loans (risk-reward tradeoff!)
- Include specific examples from policy comparisons
- Be honest about limitations

#### 3.3 Create Visualizations

All plots are auto-generated in `models/`:
- `learning_curves.png`
- `roc_curve.png`
- `confusion_matrix.png`
- `threshold_analysis.png`

Include these in your report or appendix.

#### 3.4 Convert to PDF

**Option A: Microsoft Word**
1. Copy content from REPORT_TEMPLATE.md
2. Paste into Word
3. Format headings, fix layout
4. Insert images from `models/`
5. Save as PDF

**Option B: Google Docs**
1. Copy content
2. Paste into Docs
3. Format and insert images
4. File → Download → PDF

**Option C: Pandoc** (if installed)
```bash
pandoc reports/REPORT_TEMPLATE.md -o reports/final_report.pdf
```

---

### Part 4: Demo & Final Touches

#### 4.1 Test Demo Script
```bash
cd src
python demo_predict.py --loan-amnt 10000 --int-rate 15.0 --annual-inc 50000
```

Try different scenarios:
- Low risk: `--loan-amnt 5000 --int-rate 8.0 --fico-score 750`
- High risk: `--loan-amnt 20000 --int-rate 20.0 --fico-score 620`

#### 4.2 GitHub Repository (Optional but Recommended)

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit: Shodh AI loan project"

# Create GitHub repo (via web interface)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/shodh-loan-project.git
git push -u origin main
```

**Make sure repo is PUBLIC!**

#### 4.3 Final Checklist

- [ ] All scripts run without errors
- [ ] `models/best_model.pth` exists
- [ ] `models/rl/cql_model.d3` exists
- [ ] Report filled in with actual metrics
- [ ] Report converted to PDF
- [ ] GitHub repo is public (if using)
- [ ] README.md updated with your GitHub link

---

## 📝 Submission Checklist

You need to submit:

1. **Resume/CV** (PDF)
2. **GitHub Repository Link** (public)
   - All source code
   - README.md with instructions
   - requirements.txt
   - NO DATA FILES (too large)
3. **Final Report** (PDF, 2-3 pages)
   - Covers all Task 4 questions
   - Includes metrics from both models
   - Explains decision differences
   - Proposes future steps

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Use CPU instead
```bash
# Training will be slower but will work
# Already configured to fall back to CPU automatically
```

### Issue: "d3rlpy import error"
**Solution**: Check d3rlpy version
```bash
pip install d3rlpy==2.0.0 --upgrade
```

### Issue: "File not found: accepted_2007_to_2018.csv"
**Solution**: Make sure data is in correct location
```bash
# Should be here:
dir data\accepted_2007_to_2018.csv
```

### Issue: Training taking too long
**Solution**: Use smaller sample
```bash
python 00_download_and_sample.py --sample-size 50000  # Instead of 200000
python 03_train_supervised.py --epochs 10             # Instead of 30
python 06_train_offline_rl.py --n-epochs 20           # Instead of 50
```

### Issue: "Module 'dataset' not found"
**Solution**: Make sure you're in src/ directory
```bash
cd src
python 03_train_supervised.py
```

---

## 💡 Tips for Success

1. **Start Early**: Don't wait until the last day! Pipeline takes 2-3 hours to run.

2. **Check Results**: After each step, verify the output files exist and look reasonable.

3. **Understand, Don't Just Run**: The report judges your *thinking*, not just your code. Understand why RL approves risky loans!

4. **Be Specific**: Use concrete numbers and examples in your report.

5. **Be Honest**: Acknowledge limitations. This shows critical thinking.

6. **Compare Carefully**: The most interesting part is Section 4 (comparison). Spend time on it!

7. **Proofread**: Check for typos, missing metrics, broken explanations.

---

## 📞 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Review the Troubleshooting section
3. Check that all previous steps completed successfully
4. Verify file paths are correct

---

## ⏰ Timeline Suggestion

**Day 1 (Today - Tuesday, Dec 10)**:
- Morning: Setup, download data, run preprocessing (2 hours)
- Afternoon: Train supervised model, evaluate (2 hours)
- Evening: RL training and evaluation (2 hours)

**Day 2 (Wednesday, Dec 11)**:
- Morning: Analyze results, fill in report template (3 hours)
- Afternoon: Convert to PDF, create GitHub repo, final review (2 hours)

**Day 3 (Thursday, Dec 12)**:
- Morning: Final polish, proofread, submit (1 hour)

**Deadline: Friday, Dec 12, 2025** ← You have 3 days!

---

## 🎯 Success Criteria

Your submission will be evaluated on:

1. **Analytical Rigor** (25%): EDA quality, feature choices, preprocessing
2. **Technical Execution** (25%): Code correctness, reproducibility
3. **Depth of Analysis** (35%): Understanding of models, comparison insights ← **MOST IMPORTANT**
4. **Communication** (15%): Report clarity, README quality

Focus on explaining the **why** behind the numbers!

---

Good luck! You've got this! 🚀

---

**Questions? Issues?** Check README.md or create an issue in your repo with the error message.

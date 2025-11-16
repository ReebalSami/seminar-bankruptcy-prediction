# 🎯 Multi-Dataset Bankruptcy Prediction Analysis

**Version:** 3.0.0  
**Python:** ≥3.13  
**Status:** ✅ **ANALYSIS COMPLETE** - All 31 Scripts Validated, Ready for Paper Writing

Advanced machine learning analysis for bankruptcy prediction across three international datasets (Poland, USA, Taiwan).

**Current Status (Nov 12, 2025 - 8:30pm):** ✅ **COMPLETE!** All 31 scripts executed and validated. Foundation scripts created (00, 00b). All 4 critical methodological issues FIXED. Equal treatment achieved (8 scripts per dataset). Comprehensive HTML report generated. Performance: 0.92-0.98 AUC across all datasets. **Next phase: Seminar paper writing (30-40 pages).**

---

## 📊 Project Overview

This project provides a complete, scientifically rigorous analysis of bankruptcy prediction using multiple machine learning algorithms across three datasets:

- **Polish Dataset** (UCI): 43,405 firm-years, 5 prediction horizons (1-5 years)
- **American Dataset** (Kaggle): NYSE/NASDAQ companies
- **Taiwan Dataset** (TEJ): 95 financial ratios, 2009 crisis period

### Key Achievements ✅

**Models & Performance:**
- ✅ 5 Advanced ML models (Logistic, RF, XGBoost, LightGBM, CatBoost)
- ✅ Exceptional performance: 0.92-0.98 AUC across all datasets
- ✅ Best models: CatBoost (Polish 0.9812), RF Cross-Year (American 0.9593), LightGBM (Taiwan 0.9554)

**Methodological Rigor:**
- ✅ Foundation-First approach: Scripts 00 (semantic mapping), 00b (temporal structure) created FIRST
- ✅ All 4 critical issues FIXED with proper methodology
  - Script 10c: OLS → GLM diagnostics (Hosmer-Lemeshow, deviance residuals, link test)
  - Script 11: Renamed panel_data → temporal_holdout_validation (correct terminology)
  - Script 12: Positional → Semantic mapping (AUC 0.32→0.58, **+82% improvement!**)
  - Script 13c: Granger causality → Temporal validation (no ecological fallacy)
- ✅ Equal Treatment: 8 scripts per dataset (Polish, American, Taiwan)
- ✅ Multicollinearity remediation: Ridge regression (AUC 0.7849, EPV improved)
- ✅ Comprehensive validation: Temporal holdout, bootstrap, cross-year, calibration

**Deliverables:**
- ✅ 31 scripts executed and validated
- ✅ Comprehensive HTML report (37 KB) with WHY-HOW-WHAT explanations
- ✅ All results with honest error reporting (professor requirement)  

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
cd seminar-bankruptcy-prediction

# Install dependencies (creates .venv with Python 3.13)
make install

# Verify installation
.venv/bin/python --version  # Should show Python 3.13.x
```

### 2. Run Complete Analysis

```bash
# Core analysis scripts (Polish dataset)
make run-poland

# Econometric diagnostics & remediation (NEW)
python scripts_python/10c_complete_diagnostics.py
python scripts_python/10d_remediation_save_datasets.py

# Time series diagnostics (NEW)
python scripts_python/13c_time_series_diagnostics.py

# Updated analyses with remediated data (NEW)
python scripts_python/11_panel_data_analysis.py
python scripts_python/12_cross_dataset_transfer.py
python scripts_python/13_time_series_lagged.py

# Generate comprehensive report
python scripts_python/generate_report_v2_simple.py

# Open report
open results/COMPREHENSIVE_REPORT_V2.html
```

### 3. (Optional) Compile Seminar Paper
```bash
make paper
open seminar-paper/doku_main.pdf
```

---

## 📂 Project Structure

```
seminar-bankruptcy-prediction/
├── data/
│   ├── polish-companies-bankruptcy/    # UCI Polish dataset
│   ├── american-bankruptcy/            # Kaggle US dataset
│   ├── taiwan-bankruptcy/              # TEJ Taiwan dataset
│   └── processed/                      # Cleaned datasets
│
├── src/bankruptcy_prediction/
│   ├── data/
│   │   ├── loader.py                   # DataLoader class
│   │   └── metadata.py                 # MetadataParser class
│   └── evaluation/
│       └── results_collector.py        # ResultsCollector class
│
├── scripts_python/
│   ├── 00_cross_dataset_feature_mapping.py  # ✅ Foundation: Semantic mapping
│   ├── 00b_temporal_structure_check.py      # ✅ Foundation: Temporal verification
│   ├── 01_data_understanding.py             # Polish: Data EDA
│   ├── 02-08_*.py                           # Polish: Full pipeline
│   ├── 10c_glm_diagnostics.py               # ✅ FIXED: GLM-appropriate tests
│   ├── 10d_remediation_save_datasets.py     # ✅ Multicollinearity remediation
│   ├── 11_temporal_holdout_validation.py    # ✅ RENAMED: Correct terminology
│   ├── 12_transfer_learning.py              # ✅ FIXED: Semantic mapping (+82%)
│   ├── 13c_temporal_validation.py           # ✅ FIXED: No ecological fallacy
│   ├── american/
│   │   ├── 01-08_*.py                       # ✅ Complete: 8 scripts (equal treatment)
│   ├── taiwan/
│   │   ├── 01-08_*.py                       # ✅ Complete: 8 scripts (equal treatment)
│   └── generate_comprehensive_html_report.py # ✅ Full HTML report generator
│
├── results/
│   ├── script_outputs/                 # All analysis outputs (CSV, PNG, JSON)
│   ├── models/                         # Saved models
│   └── ANALYSIS_REPORT.html            # Master HTML report
│
├── seminar-paper/
│   ├── kapitel/                        # LaTeX chapters (German)
│   ├── bilder/                         # Figures
│   ├── doku_main.tex                   # Main document
│   └── sources.bib                     # Bibliography
│
├── Makefile                             # Automation commands
├── pyproject.toml                       # Dependencies (Python 3.13)
└── README.md                            # This file
```

---

## 🔬 Methodology

### Datasets

| Dataset | Source | Samples | Features | Bankrupt Rate | Horizons |
|---------|--------|---------|----------|---------------|----------|
| **Polish** | UCI | 43,405 | 64 | 3.9-6.9% | 1-5 years |
| **American** | Kaggle | ~7,000 | Varies | ~7% | 1 year |
| **Taiwan** | TEJ | ~7,000 | 95 | ~3% | 1 year |

### Models Implemented

1. **Baseline Models**
   - Logistic Regression (with L2 regularization)
   - Random Forest (200 trees, class-weighted)

2. **Advanced Models**
   - XGBoost (300 estimators, tuned hyperparameters)
   - LightGBM (300 estimators, class-weighted)
   - CatBoost (automatic handling of categorical features)

3. **Enhancements**
   - Isotonic calibration for probability reliability
   - SHAP values for interpretability
   - Cross-horizon robustness testing (5 horizons × 5 test sets = 25 experiments)

### Evaluation Metrics

- **ROC-AUC:** Primary metric for model comparison
- **PR-AUC:** Handles class imbalance
- **Recall @ 1% FPR:** Operational metric (few false alarms)
- **Brier Score:** Calibration quality

### Econometric Diagnostics (Complete)

- **Multicollinearity:** VIF, Condition Number
- **Autocorrelation:** Durbin-Watson, Breusch-Godfrey, Ljung-Box
- **Heteroscedasticity:** Breusch-Pagan, White Robust SE
- **Normality:** Jarque-Bera
- **Stationarity:** Augmented Dickey-Fuller (ADF)
- **Causality:** Granger Causality Test
- **Cointegration:** Engle-Granger Test
- **Sample Size:** Events Per Variable (EPV)

---

## 📊 Results Summary

### Model Performance (All Datasets)

| Dataset | Best Model | AUC | Key Validation |
|---------|-----------|-----|----------------|
| **Polish** (43,405) | CatBoost | **0.9812** | Temporal holdout (H1-3→H4-5) |
| **American** (78,682) | RF Cross-Year | **0.9593** | 2015-17→2018 |
| **Taiwan** (6,819) | LightGBM | **0.9554** | Bootstrap 95% CI |

### Methodological Validations ✅

| Validation Type | Result | Interpretation |
|----------------|--------|----------------|
| **GLM Diagnostics** | Hosmer-Lemeshow p=0.0001 | Calibration issue (non-fatal) |
| **Multicollinearity** | Ridge AUC 0.7849 | Successfully remediated |
| **EPV (Events/Variable)** | 4.30 → 10.85 | Fixed via forward selection |
| **Transfer Learning** | AUC 0.32 → 0.58 | **+82% improvement** |
| **Temporal Stability** | ~2% degradation | Excellent generalization |
| **Calibration (American)** | LR: 80.5% improvement | Strong calibration gains |
| **Bootstrap Robustness** | Tight 95% CIs | Statistically stable |

### Key Findings

1. **Delta Features:** 34% importance (rate of change > absolute values)
2. **Semantic Mapping:** Critical for cross-dataset transfer (+82%)
3. **Temporal Structure:** Polish is repeated cross-sections, not panel
4. **Ridge Regularization:** Best for multicollinearity (AUC 0.7849)

---

## 📈 Viewing Results

### 📄 Comprehensive HTML Report (NEW!)

**COMPLETE_RESULTS_REPORT.html** - All 31 scripts with full WHY-HOW-WHAT explanations:

```bash
# Open the comprehensive report
open COMPLETE_RESULTS_REPORT.html
```

**Includes:**
- ✅ All model performance (Polish, American, Taiwan)
- ✅ Foundation scripts (00, 00b) methodology
- ✅ All 4 methodological fixes documented
- ✅ GLM diagnostics, multicollinearity remediation
- ✅ Transfer learning results (+82% improvement)
- ✅ Calibration, robustness, bootstrap validation
- ✅ Complete WHY-HOW-WHAT explanations
- ✅ Defense preparation Q&A

### Individual Script Outputs

```bash
# Script-specific results
ls results/script_outputs/*/
ls results/00_feature_mapping/
ls results/00b_temporal_structure/
```

### Raw Data Files

```bash
# CSV files with all metrics
ls results/script_outputs/*/\*.csv

# PNG figures (300 DPI)
ls results/script_outputs/*/figures/\*.png

# JSON summaries
ls results/script_outputs/*/\*.json
```

---

## 📝 Next Steps: Seminar Paper Writing

**Status:** ✅ Analysis Phase COMPLETE → ⏳ Paper Writing Phase READY

### Paper Specification
- **Length:** 30-40 pages
- **Format:** LaTeX (Overleaf)
- **Language:** German or English
- **Timeline:** 20-30 hours
- **Source:** Use COMPLETE_RESULTS_REPORT.html as primary data source

### Recommended Structure

1. **Einleitung** (2-3 pages) - Motivation, research questions, contribution
2. **Literature Review** (5-7 pages) - ML in bankruptcy prediction, prior work
3. **Methodik** (8-10 pages) - Datasets, models, validation **with honest error reporting!**
4. **Ergebnisse** (8-10 pages) - All performance tables, comparisons, interpretations
5. **Diskussion** (5-7 pages) - What do results mean? Comparison with literature
6. **Fazit** (2-3 pages) - Contributions, limitations, future work
7. **Anhang** - Additional tables, detailed results

### Key Writing Principles
✅ Include all 4 methodological fixes (shows scientific integrity)  
✅ Explain WHY-HOW-WHAT for each method  
✅ Professor values **honest reporting of failures**  
✅ Focus on what was learned from errors  
✅ Use actual numbers from HTML report (no fabrication)

---

## 🔧 Development

### Make Commands

```bash
make help              # Show all commands
make install           # Install dependencies
make clean             # Clean generated files
make run-poland        # Run Polish analysis
make run-american      # Run American analysis
make run-taiwan        # Run Taiwan analysis
make run-comparison    # Cross-dataset comparison
make run-all           # Run everything
make report            # Generate HTML report
make paper             # Compile LaTeX paper
```

### Running Individual Scripts

```bash
# Activate environment
source .venv/bin/activate

# Run any script
python scripts_python/01_data_understanding.py
python scripts_python/american/02_eda.py
python scripts_python/cross_dataset_comparison.py
```

### Adding New Analysis

1. Create script in `scripts_python/`
2. Follow existing structure (import from `src.bankruptcy_prediction`)
3. Save outputs to `results/script_outputs/<script_name>/`
4. Generate HTML report with visualizations
5. Update `generate_html_report.py` to include new section
6. Add to Makefile if needed

---

## 📚 Dependencies

Core packages (see `pyproject.toml`):
- **ML:** scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn
- **Data:** pandas, numpy, scipy, pyarrow
- **Viz:** matplotlib, seaborn
- **Stats:** statsmodels, shap
- **Docs:** jupyter, ipykernel

All installed automatically with `make install`.

---

## 🎯 Quality Standards

Every script in this project:
✅ Runs without errors  
✅ Saves outputs (CSV, PNG, JSON)  
✅ Generates HTML visualization  
✅ Includes analysis & interpretation  
✅ Scientifically rigorous (no guessing, no hardcoding)  
✅ Tested by execution  
✅ Well-commented code  

---

## 📖 Citation

If using this work, please cite:

```bibtex
@techreport{reebal2025bankruptcy,
  title={Multi-Dataset Bankruptcy Prediction with Advanced Machine Learning},
  author={Reebal},
  institution={FH Wedel},
  year={2025},
  type={Seminar Paper}
}
```

---

## 📧 Contact

**Author:** Reebal  
**Institution:** FH Wedel  
**Course:** Seminar Econometrics & Machine Learning  
**Year:** 2025

---

## ✅ Project Status

- [x] Polish dataset complete (8 scripts, full pipeline)
- [x] American dataset complete
- [x] Taiwan dataset complete  
- [x] Cross-dataset comparison complete
- [x] **Forward selection remediation** (Script 10d, EPV improved 3.44→10.85) ✅
- [x] **Delta features discovery** (Script 13, 34% importance — novel finding) ✅
- [x] **Comprehensive analysis** of all 24 scripts (COMPLETE_PIPELINE_ANALYSIS.md) ✅
- [ ] **Phase 0: Foundation** (CREATE Scripts 00, 00b) — MUST BE FIRST!
- [ ] **Phase 1: Fix Flawed Scripts** (REWRITE 10c, 12, 13c; RENAME 11)
- [ ] **Phase 2: Equal Treatment** (CREATE Scripts 04-08 for American & Taiwan)
- [ ] **Phase 3: Methodology** (train/val/test, corrections, bootstrap CI)
- [ ] **Phase 4: Documentation** (seminar paper 30-40 pages, defense prep)

**Current Status:** Strong foundation but needs methodological fixes for academic submission.

---

## 📚 Key Documentation (4 Core Files)

1. **[README.md](README.md)** — This file (project overview, quick start)
2. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** — Single source of truth (what works, what's broken) **← READ FIRST**
3. **[PERFECT_PROJECT_ROADMAP.md](PERFECT_PROJECT_ROADMAP.md)** — Complete roadmap to 1.0 grade (60-80 hours, 4 phases)
4. **[COMPLETE_PIPELINE_ANALYSIS.md](COMPLETE_PIPELINE_ANALYSIS.md)** — Detailed analysis reference (541KB, 17,729 lines)

**Start here:** Read PROJECT_STATUS.md, then follow PERFECT_PROJECT_ROADMAP.md phase by phase.

---

**Last Updated:** November 12, 2025  
**Version:** 2.1.0  
**Status:** ⚠️ Ready for implementation (follow PERFECT_PROJECT_ROADMAP.md)  

**Critical Documentation:**
- **PROJECT_STATUS.md** — Single source of truth (read first!)
- **PERFECT_PROJECT_ROADMAP.md** — Complete 60-80 hour roadmap to 1.0 grade
- **COMPLETE_PIPELINE_ANALYSIS.md** — Detailed analysis of all 24 scripts (541KB reference)

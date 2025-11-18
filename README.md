# Bankruptcy Prediction - Seminar Project

**Institution:** FH Wedel  
**Semester:** WS 2024/25  
**Topic:** Entwicklung eines Frühwarnsystems für Unternehmenskrisen mit Hilfe maschinellen Lernens  
**Goal:** German grade 1.0 (excellent)

---

## Project Overview

**Research Focus:**
- Early warning indicators for corporate bankruptcy
- Predictive analytics using financial ratios
- Comparison: Random Forests vs Logistic Regression
- Multi-horizon prediction (1-5 years ahead)

**Dataset:** Polish Companies Bankruptcy (Kaggle)
- 43,405 observations
- 64 financial ratio features (A1-A64)
- 5 prediction horizons (H1-H5)
- Target: Bankruptcy within horizon period

---

## Current Status: Foundation Phase Complete ✅

### **Phase 00: Foundation (100% DONE)**

| Script | Purpose | Status |
|--------|---------|--------|
| 00a | Dataset overview, feature mapping | ✅ Complete |
| 00b | Feature analysis, category distribution | ✅ Complete |
| 00c | Temporal structure, bankruptcy trends | ✅ Complete |
| 00d | Data quality assessment | ✅ Complete |

**Key Findings:**
- ✅ 64 features: 6 categories (Profitability:20, Leverage:17, Activity:15, Liquidity:10, Size:1, Other:1)
- ⚠️ ALL 64 features have missing values (max: A37 at 43.7%)
- ⚠️ 401 duplicate rows (200 pairs) - assumed data entry errors
- ⚠️ ALL 64 features have outliers (0.07%-15.5%, mean: 5.4%)
- ⚠️ **Bankruptcy rate increases 80%: H1 (3.86%) → H5 (6.94%)**
- ✅ Data structure: Repeated cross-sections (NOT panel data)
- ✅ **Decision Made:** Horizon-specific modeling (5 separate models)

---

### **Phase 01: Data Preparation (100% DONE)**

| Script | Purpose | Status |
|--------|---------|--------|
| 01a | Remove duplicates | Complete |
| 01b | Outlier treatment (winsorization) | Complete |
| 01c | Missing value imputation | Complete |

**Results:**
- 43,004 observations (401 duplicates removed)
- 0% missing values (imputation quality: 98.2/100)
- Outliers dampened via winsorization (1st/99th percentiles)
- Clean dataset ready for exploratory analysis

**Output File:** `data/processed/poland_imputed.parquet`

---

## Next Phase: Exploratory Data Analysis (Phase 02)

**Planned Activities:**
- Distribution analysis per horizon (H1-H5)
- Univariate feature analysis (t-tests, effect sizes)
- Correlation matrices and multicollinearity checks
- Feature importance rankings

**Research-Backed Sequence:**
```
Remove duplicates → Treat outliers → Impute missing values → Scale → Split
```

**Why this order:** "When outliers are removed or treated and missing values accurately imputed, the correlations among predictors become more realistic" (Number Analytics, 2024)

---

## Project Structure

```
seminar-bankruptcy-prediction/
├── scripts/
│   ├── 00_foundation/       # Dataset understanding (COMPLETE)
│   ├── 01_data_preparation/ # Cleaning & preprocessing (NEXT)
│   ├── 02_exploratory/      # EDA on cleaned data
│   ├── 03_multicollinearity/# VIF analysis & feature reduction
│   ├── 04_feature_selection/# Feature importance & selection
│   └── 05_model_evaluation/ # Modeling & evaluation
│
├── results/
│   └── 00_foundation/       # Excel, HTML, PNG outputs
│
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned data
│
├── docs/
│   └── 00_FOUNDATION_CRITICAL_FINDINGS.md  # Complete review
│
└── src/bankruptcy_prediction/  # Shared utilities
    ├── data/                   # Data loaders
    ├── features/               # Feature engineering
    └── utils/                  # Logging, config
```

---

## Key Methodological Decisions

### ✅ **What We Got Right**

1. **Analysis-First Approach:** Understand data before preprocessing
2. **Foundation Phase:** Complete characterization of dataset
3. **Evidence-Based:** Research citations for all decisions
4. **Honest Reporting:** Document limitations and assumptions

### ⚠️ **Critical Issues Identified**

1. **Duplicate Nature Unknown:**
   - 401 exact duplicates (all 68 columns identical)
   - NO company ID → can't determine if same company or error
   - **Assumption:** Data entry errors → remove in Phase 01

2. **Horizon Heterogeneity:**
   - Bankruptcy rate: 3.86% (H1) → 6.94% (H5) = **80% increase**
   - Foundation analyzed ALL horizons combined
   - **Decision needed:** Separate models OR pooled with horizon feature

3. **Incomplete Initial Analysis:**
   - Script 00d v1: Only 10/64 features for outliers ❌
   - Script 00d v2: ALL 64 features analyzed ✅

---

## Running the Project

### Setup
```bash
make install  # Activate venv and sync dependencies
```

### Execute Foundation Scripts
```bash
python scripts/00_foundation/00a_polish_dataset_overview.py
python scripts/00_foundation/00b_polish_feature_analysis.py
python scripts/00_foundation/00c_polish_temporal_structure.py
python scripts/00_foundation/00d_polish_data_quality.py
```

### View Results
```bash
open results/00_foundation/00a_polish_overview.html
open results/00_foundation/00d_data_quality.xlsx
```

---

## Documentation

**Core Files (ONLY):**
1. **README.md** (this file) - Project overview
2. **docs/PROJECT_STATUS.md** - Current state tracker
3. **PERFECT_PROJECT_ROADMAP.md** - Phase-by-phase plan
4. **COMPLETE_PIPELINE_ANALYSIS.md** - Detailed methodology review

**Archived:**
- `archive/docs_old/` - Historical documentation

---

## Research Methodology

**Preprocessing Pipeline Order (Evidence-Based):**
1. Remove duplicates (prevent leakage)
2. Treat outliers (3×IQR winsorization)
3. Impute missing values (passive imputation for ratios)
4. Scale features (z-score normalization)
5. Split data (temporal holdout: H1-H3 / H4 / H5)

**Then:**
6. Calculate VIF (requires complete data)
7. Remove multicollinear features (VIF > 10)
8. Feature selection (forward/backward, importance)
9. Model training (Logistic, Random Forest, XGBoost)
10. Evaluation (ROC-AUC, PR-AUC, calibration)

**Critical:** VIF analysis MUST come AFTER imputation (research-backed)

---

## Professor's Criteria (Grade 1.0)

✅ **Methodology:** Evidence-based, research citations  
✅ **Honesty:** Document failures and assumptions  
✅ **Completeness:** 30-40 pages, all phases covered  
✅ **Econometrics:** Proper GLM diagnostics (not OLS)  
✅ **Validation:** Temporal holdout, no data leakage  

---

## References

1. Number Analytics (2024). "VIF Strategies: Reducing Multicollinearity"
2. Von Hippel (2013). "Multiple imputation for ratios." *Statistics in Medicine*
3. Coats & Fant (1993). "Bankruptcy prediction across time horizons"
4. Feature-engine documentation. "Missing Data Imputation"

---

**Status:** Foundation phase validated. Ready for Phase 01 after strategy decision.

# Bankruptcy Prediction - Seminar Project

**Institution:** FH Wedel  
**Semester:** WS 2024/25  
**Topic:** Entwicklung eines Frühwarnsystems für Unternehmenskrisen mit Hilfe maschinellen Lernens

---

## Project Overview

**Research Focus:**
- Early warning system for corporate bankruptcy
- Multi-horizon prediction (1-5 years before bankruptcy)
- Machine learning methods for financial distress prediction

**Dataset:** Polish Companies Bankruptcy Data
- 43,004 observations (after duplicate removal)
- 64 financial ratio features
- 5 prediction horizons (H1-H5)
- Class distribution: 4.84% bankruptcy rate

---

## Project Results

### Pipeline Phases

**Phase 00-01:** Foundation & Data Preparation
- Dataset: 43,004 observations (401 duplicates removed), 64 financial ratios
- Horizon-specific modeling approach (H1-H5)
- Class imbalance: 4.84% bankruptcy rate
- Outlier treatment and missing value imputation completed

**Phase 02-03:** EDA & Multicollinearity Analysis  
- VIF-based feature reduction per horizon
- Temporal structure analysis across 5 horizons
- Correlation and distribution analysis

**Phase 04:** Feature Selection
- Automated pipeline: Filter methods (Spearman, MI, ANOVA-F)
- Wrapper methods (RFECV with cross-validation)
- Embedded methods (Lasso, Elastic Net, Ridge, Random Forest)
- Consensus approach with EPV guardrails (≥10 events per variable)
- Final feature sets: 9 features per horizon (H1-H5)

**Phase 05-06:** Modeling & Evaluation
- Automated modeling pipeline with 5-fold Stratified CV
- Class balancing via class_weight parameter
- Best models: Soft-Voting Ensemble (H1: 0.796 AUC), Random Forest (H2-H5: 0.780-0.864 AUC)
- Evaluation metrics: ROC-AUC and PR-AUC

---

## Project Structure

```
seminar-bankruptcy-prediction/
├── config/
│   └── project_config.yaml   # Centralized configuration (datasets, CV, models)
│
├── scripts/
│   ├── 00_foundation/        # Dataset understanding & foundation review
│   ├── 01_data_preparation/  # Duplicate removal, outlier treatment, imputation
│   ├── 02_exploratory_analysis/ # EDA, correlation, temporal analysis
│   ├── 03_multicollinearity/ # VIF analysis & feature reduction
│   ├── 04_feature_selection/ # Filter, wrapper, embedded methods + consensus
│   ├── 05_modeling/          # Base models, extra models (ensembles)
│   ├── 06_model_evaluation/  # Aggregation, evaluation plots, metrics
│   └── paper_helper/         # LaTeX table/figure generation scripts
│
├── results/
│   ├── 00_foundation/        # Foundation review outputs
│   ├── 01_data_preparation/  # Cleaned datasets, outlier reports
│   ├── 02_exploratory_analysis/ # EDA figures, correlation matrices
│   ├── 03_multicollinearity/ # VIF analysis results
│   ├── 04_feature_selection/ # Selected feature sets (H1-H5)
│   ├── 05_modeling/          # Model training outputs (base + extra)
│   └── 06_model_evaluation/  # Final metrics, ROC/PR curves
│
├── data/
│   ├── polish-companies-bankruptcy/ # Original Polish bankruptcy data
│   └── processed/            # Cleaned, imputed data + feature sets
│
├── logs/                     # Execution logs organized by phase
│   ├── 00_foundation/        # Phase 00 logs
│   ├── 01_data_preparation/  # Phase 01 logs
│   ├── 02_exploratory_analysis/ # Phase 02 logs
│   ├── 03_multicollinearity/ # Phase 03 logs
│   ├── 04_feature_selection/ # Phase 04 logs (base + nested)
│   └── 05_modeling/          # Phase 05 logs (base + extra)
│
├── seminar-paper/
│   ├── doku_main.tex         # Main LaTeX paper
│   ├── kapitel/              # Paper chapters
│   ├── tables/               # Auto-generated LaTeX tables
│   ├── figures/              # Auto-generated figures
│   └── addendum/             # Post-submission nachreichung
│
├── src/bankruptcy_prediction/ # Shared utilities (Phases 00-03 only)
│   └── utils/                # Configuration, logging, metadata, target utils
│
└── Makefile                  # Full pipeline automation
```

---

## Methodology

### Key Decisions

1. **Horizon-Specific Models:** Separate models for H1-H5 to account for heterogeneous prediction tasks
2. **Consensus-Based Feature Selection:** Multi-method integration (Filter + Wrapper + Embedded)
3. **EPV Guardrails:** Minimum 10 events per variable enforced in consensus step
4. **Automated Pipeline:** Complete workflow from raw data to final paper via Makefile
5. **Class Balancing:** Stratified CV with class_weight='balanced' parameter

### Data Characteristics

- **Total observations:** 43,004 (401 duplicates removed)
- **Features:** 64 financial ratios (A1-A64)
- **Class distribution:** 4.84% bankruptcy (highly imbalanced)
- **Horizons:** H1 (1 year) to H5 (5 years) before bankruptcy
- **Final feature sets:** 9 consensus features per horizon

---

## Reproducibility

### Configuration

All parameters are centralized in `config/project_config.yaml`:
- Dataset paths and feature definitions
- Class balancing strategy (SMOTE, class_weight)
- CV strategy (n_splits=5, stratified)
- Feature selection thresholds
- Model hyperparameters
- Random seed (42)

### Pipeline Execution

**Setup:**
```bash
make install  # Creates .venv with Python 3.13 and installs dependencies
```

**Complete Pipeline (All Phases):**
```bash
make phase01  # Data preparation (duplicates, outliers, imputation)
make phase02  # Exploratory analysis (distributions, tests, correlations)
make phase03  # Multicollinearity analysis (VIF)
make phase04  # Feature selection (filter, wrapper, embedded, consensus)
make phase05-modeling        # Base models (LR, RF)
make phase05-modeling-extra  # Extra models (GB, ET, SVC, Soft-Voting)
make phase05-eval            # Aggregate metrics and plots
```

**Paper Generation:**
```bash
make phase04-tables  # Generate Phase 04 feature selection tables
make phase05-tables  # Generate Phase 05/06 modeling tables
make paper           # Compile LaTeX paper: seminar-paper/doku_main.pdf
```

**Nested CV Validation (Post-Submission):**
```bash
make phase04d-consensus-nested     # Nested feature selection
make phase05-modeling-nested       # Nested base models
make phase05-modeling-extra-nested # Nested extra models
make phase06-eval-nested           # Nested evaluation
make addendum                      # Compile nachreichung PDF
```

### Individual Script Execution

All scripts can be run individually via Python:
```bash
.venv/bin/python scripts/04_feature_selection/04a_filter_methods.py
.venv/bin/python scripts/05_modeling/05_modeling_train_evaluate.py
# etc.
```

Parameters are loaded from `config/project_config.yaml` ensuring consistency.

### View Results
- **Paper:** `seminar-paper/doku_main.pdf`
- **Data quality:** `results/00_foundation/*.html`
- **Feature selection:** `results/04_feature_selection/`
- **Model metrics:** `results/06_model_evaluation/05_ALL_model_eval.xlsx`

---

## References

1. Number Analytics (2024). "VIF Strategies: Reducing Multicollinearity"
2. Von Hippel (2013). "Multiple imputation for ratios." *Statistics in Medicine*
3. Coats & Fant (1993). "Bankruptcy prediction across time horizons"
4. Nogueira et al. (2018). "Stability of feature selection algorithms." *Data Mining and Knowledge Discovery*

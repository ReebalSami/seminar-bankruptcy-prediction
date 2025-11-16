# Phase 01: Data Preparation - Handover Prompt for New Chat

**Copy this entire prompt to the new chat to continue with Phase 01**

---

## Project Context

I am working on a seminar paper (FH Wedel, WS 2024/25) about developing an early warning system for corporate bankruptcy using machine learning. The project compares ML models (Random Forest, XGBoost) against classical logistic regression on Polish company bankruptcy data.

**Target:** German grade 1.0 (excellent)  
**Professor values:** Honest reporting of failures, perfect methodology, econometric validation

---

## Phase 00: Foundation - COMPLETE ✅

I have successfully completed the Foundation Phase with 4 scripts (00a-00d) that analyzed the Polish Companies Bankruptcy dataset:

### Key Findings from Phase 00:
1. **Dataset:** 43,405 observations, 64 financial ratio features (A1-A64), 5 prediction horizons (H1-H5)
2. **Data structure:** Repeated cross-sections (NO company ID, not panel data)
3. **Critical finding:** Bankruptcy rate increases 80% from H1 (3.86%) to H5 (6.94%)
4. **Decision made:** Horizon-specific modeling (5 separate models, one per horizon)
5. **Data quality issues identified:**
   - ALL 64 features have missing values (worst: A37 at 43.7%)
   - 401 exact duplicates (200 pairs) - assumed data entry errors
   - ALL 64 features have outliers (2.1%-15.5% per feature)
   - Multicollinearity expected (inverse pairs, shared denominators)

### Outputs Created:
- **Code:** 4 Python scripts in `scripts/00_foundation/`
- **Results:** 12 files (Excel, HTML, PNG) in `results/00_foundation/`
- **Paper:** Chapter 3 (~17 pages in German) in `seminar-paper/kapitel/03_Daten_und_Methodik.tex`
- **Docs:** `docs/PROJECT_STATUS.md` and `docs/00_FOUNDATION_CRITICAL_FINDINGS.md`

---

## Phase 01: Data Preparation - TO IMPLEMENT NOW

### Objective:
Implement the data preparation pipeline based on evidence-based research, converting the raw dataset into analysis-ready horizon-specific datasets.

### Research-Backed Sequence:
```
1. Remove duplicates (prevent data leakage)
   ↓
2. Treat outliers (prevent bias in imputation)
   ↓
3. Impute missing values (passive imputation for ratios)
   ↓
4. Split into horizons (5 separate datasets)
   ↓
5. Scale features (z-score normalization per horizon)
```

**Research citations:**
- **Goodfellow et al. (2016):** Duplicates must be removed BEFORE train/test split to prevent data leakage
- **Number Analytics (2024):** Outlier treatment BEFORE imputation for realistic correlations
- **von Hippel (2013):** Passive imputation for financial ratios (impute numerator/denominator separately after log-transform)

---

## Phase 01 Scripts to Create (4 scripts)

### Script 01a: Remove Duplicates
**File:** `scripts/01_data_preparation/01a_remove_duplicates.py`

**Inputs:**
- `data/processed/poland_clean_full.parquet` (43,405 observations)

**Logic:**
1. Load data
2. Identify exact duplicates (all 68 columns identical) - should find 401 rows
3. Drop duplicates, keep first instance
4. Document: how many removed, from which horizons
5. Save cleaned data

**Outputs:**
- `data/processed/poland_no_duplicates.parquet` (43,004 observations)
- `results/01_data_preparation/01a_duplicate_removal_report.xlsx` (summary stats)
- `results/01_data_preparation/01a_duplicate_removal.html` (dashboard)

**Validation:**
- Verify 401 rows removed (matching Phase 00 finding)
- Confirm no duplicates remain
- Check horizon distribution unchanged (proportionally)

---

### Script 01b: Outlier Treatment
**File:** `scripts/01_data_preparation/01b_outlier_treatment.py`

**Inputs:**
- `data/processed/poland_no_duplicates.parquet`

**Logic:**
1. Load data
2. For EACH of 64 features:
   - Calculate 1st and 99th percentile
   - Winsorize: values below P1 → P1, values above P99 → P99
3. Document: how many values affected per feature
4. Save winsorized data

**Outputs:**
- `data/processed/poland_winsorized.parquet`
- `results/01_data_preparation/01b_outlier_treatment.xlsx` (summary per feature)
- `results/01_data_preparation/01b_before_after_distributions.png` (sample features)

**Validation:**
- ~10% of values per feature should be affected (based on Phase 00 analysis)
- Variance reduced but not eliminated
- No values outside new bounds

---

### Script 01c: Missing Value Imputation
**File:** `scripts/01_data_preparation/01c_missing_value_imputation.py`

**Inputs:**
- `data/processed/poland_winsorized.parquet`
- Feature metadata with numerator/denominator info

**Logic - PASSIVE IMPUTATION (von Hippel 2013):**
1. For each ratio feature (A1-A64):
   - Identify numerator and denominator from formula
   - Log-transform numerator and denominator
   - Impute EACH separately using IterativeImputer (scikit-learn)
   - Back-transform (exp)
   - Calculate ratio = numerator / denominator
2. Special handling for A37 (43.7% missing):
   - Document imputation quality (RMSE if possible)
   - Consider flagging imputed values
3. Save imputed data

**Outputs:**
- `data/processed/poland_imputed.parquet` (0% missing values)
- `results/01_data_preparation/01c_imputation_report.xlsx` (quality metrics per feature)
- `results/01_data_preparation/01c_imputation_quality.png` (observed vs imputed distributions)

**Validation:**
- 0% missing values after imputation
- Distributions of imputed values similar to observed
- No unrealistic values (negative ratios, extreme values)
- A37: Check if imputation quality sufficient or feature should be dropped

---

### Script 01d: Create Horizon Datasets & Scale
**File:** `scripts/01_data_preparation/01d_create_horizon_datasets.py`

**Inputs:**
- `data/processed/poland_imputed.parquet`

**Logic:**
1. Split data by horizon (H1-H5) into 5 datasets
2. For EACH horizon separately:
   a. Stratified train/val/test split (60/20/20)
      - Preserve bankruptcy rate in each split
      - Random seed for reproducibility
   b. Z-score normalization:
      - Fit scaler on TRAIN set only
      - Transform train/val/test using same scaler
   c. Save 3 files per horizon (train/val/test)
3. Document final dataset sizes and bankruptcy rates

**Outputs:**
- `data/processed/horizons/H1_train.parquet` (+ val, test)
- `data/processed/horizons/H2_train.parquet` (+ val, test)
- ... (H3, H4, H5)
- `data/processed/horizons/scaling_parameters.pkl` (scaler objects for each horizon)
- `results/01_data_preparation/01d_horizon_splits_summary.xlsx`
- `results/01_data_preparation/01d_final_datasets.html`

**Validation:**
- 15 parquet files created (5 horizons × 3 splits)
- Bankruptcy rate consistent across train/val/test per horizon
- Features scaled: mean≈0, std≈1 in train sets
- No data leakage: val/test scaled using train statistics only

---

## Technical Specifications

### Environment:
- Python 3.11+ with uv package manager
- Always run: `make install` at start of session to activate .venv
- All dependencies in `pyproject.toml`

### Project Structure:
```
seminar-bankruptcy-prediction/
├── scripts/
│   ├── 00_foundation/           ✅ COMPLETE
│   └── 01_data_preparation/     ← CREATE THIS
│
├── data/processed/
│   ├── poland_clean_full.parquet          (original, 43,405 obs)
│   ├── poland_no_duplicates.parquet       (from 01a)
│   ├── poland_winsorized.parquet          (from 01b)
│   ├── poland_imputed.parquet             (from 01c)
│   └── horizons/                          (from 01d)
│       ├── H1_train.parquet
│       ├── H1_val.parquet
│       ├── H1_test.parquet
│       ├── ... (H2-H5)
│       └── scaling_parameters.pkl
│
├── results/01_data_preparation/  ← CREATE THIS
│   ├── 01a_duplicate_removal_report.xlsx
│   ├── 01b_outlier_treatment.xlsx
│   ├── 01c_imputation_report.xlsx
│   └── 01d_horizon_splits_summary.xlsx
│
├── docs/
│   ├── PROJECT_STATUS.md         (update after Phase 01)
│   └── 00_FOUNDATION_CRITICAL_FINDINGS.md
│
└── seminar-paper/
    ├── kapitel/
    │   └── 04_Datenaufbereitung.tex  (write after Phase 01 complete)
    └── sources.bib
```

### Code Standards:
1. **Use shared utilities:** Import from `src/bankruptcy_prediction/` where possible
2. **Logging:** Use `src/bankruptcy_prediction/utils/logging_setup.py`
3. **Config:** Load from `config/project_config.yaml` (no hardcoded paths)
4. **Outputs:** Generate Excel (detailed), HTML (dashboard), PNG (visualizations) for each script
5. **Documentation:** Every script has docstring explaining purpose, inputs, outputs, methodology

### Quality Requirements:
- ✅ **Evidence-based:** Cite research for each methodological choice
- ✅ **Complete:** No shortcuts, no sampling, full dataset analysis
- ✅ **Validated:** Run scripts end-to-end, verify outputs match expectations
- ✅ **Documented:** Update docs/ and seminar-paper/ after each script
- ✅ **No hallucination:** Only report what scripts actually produce

---

## User Preferences (CRITICAL - ALWAYS FOLLOW)

From user memory rules:

1. **Anti-laziness:** NO shortcuts, NO "sample-only" runs, ALWAYS full data, FIX don't DELETE
2. **Honesty:** 100% honest, 100% direct, 100% critical. No sugarcoating.
3. **Make it work:** Scripts MUST run end-to-end and produce expected results. Explain every function.
4. **No hallucination:** Never make up results. Always base on proof. Check files before claiming things.
5. **Documentation:** Update ONLY 4 core docs (README, PROJECT_STATUS, PERFECT_ROADMAP, COMPLETE_PIPELINE). No extra .md files.
6. **No one-liners:** No python -c "...", no cat <<EOF. Always create .py files.
7. **Run sequentially:** If you fix a script, re-run ALL subsequent scripts in order.
8. **Use uv + Makefile:** Start every session with `make install`

---

## Expected Deliverables for Phase 01

### Code:
- [ ] `scripts/01_data_preparation/01a_remove_duplicates.py`
- [ ] `scripts/01_data_preparation/01b_outlier_treatment.py`
- [ ] `scripts/01_data_preparation/01c_missing_value_imputation.py`
- [ ] `scripts/01_data_preparation/01d_create_horizon_datasets.py`

### Data:
- [ ] `data/processed/poland_no_duplicates.parquet` (43,004 obs)
- [ ] `data/processed/poland_winsorized.parquet` (43,004 obs, winsorized)
- [ ] `data/processed/poland_imputed.parquet` (43,004 obs, 0% missing)
- [ ] `data/processed/horizons/H1-H5_train/val/test.parquet` (15 files)
- [ ] `data/processed/horizons/scaling_parameters.pkl`

### Results:
- [ ] 4 Excel reports (one per script)
- [ ] 4 HTML dashboards
- [ ] 2+ PNG visualizations
- [ ] All validated against Phase 00 expectations

### Documentation:
- [ ] Update `docs/PROJECT_STATUS.md` (Phase 01 complete section)
- [ ] Write `seminar-paper/kapitel/04_Datenaufbereitung.tex` (~4-5 pages in German)
  - Describe passive imputation in detail (von Hippel 2013)
  - Document A37 handling (43.7% missing)
  - Explain horizon-specific scaling rationale
  - Include tables/figures from results

### Validation:
- [ ] All 4 scripts run without errors
- [ ] Final datasets: 15 parquet files (5 horizons × 3 splits)
- [ ] Bankruptcy rates preserved in splits
- [ ] Features scaled correctly (mean≈0, std≈1)
- [ ] No data leakage confirmed

---

## How to Proceed

1. **Start session:** Run `make install` to activate environment
2. **Create scripts one by one** in order (01a → 01b → 01c → 01d)
3. **For each script:**
   - Write complete implementation
   - Include docstrings and inline comments
   - Generate professional outputs (Excel, HTML, PNG)
   - Run end-to-end and verify results
   - Show me key findings
4. **After all 4 scripts complete:**
   - Update `docs/PROJECT_STATUS.md`
   - Write Chapter 04 in German for seminar paper
   - Validate entire pipeline (check all 15 final files)
5. **Summary:** Provide overview of what was accomplished

---

## Important Notes

- **A37 decision:** Try passive imputation first. If quality poor (RMSE high, unrealistic values), document and consider dropping feature.
- **Horizon splits:** Each horizon gets own train/val/test. Do NOT pool horizons (heterogeneity documented in Phase 00).
- **Scaling:** Fit scaler on train, apply to val/test. Store scaler objects for later use (Phase 05 predictions).
- **Random seed:** Use consistent seed (e.g., 42) for reproducibility.
- **Validation:** Cross-check final observation counts against Phase 00 (should lose only 401 to duplicates).

---

## Questions I May Have

- **Q:** Should I impute before or after splitting horizons?
  **A:** BEFORE. Imputation uses all data for better estimates. Split comes after imputation is complete.

- **Q:** What imputation method exactly?
  **A:** Passive imputation per von Hippel (2013):
    1. Extract numerator/denominator formulas from metadata
    2. Log-transform both
    3. Impute each separately (IterativeImputer with default settings)
    4. Back-transform (exp)
    5. Calculate ratio
  
- **Q:** What if A37 imputation quality is bad?
  **A:** Document the quality metrics. If RMSE very high or values unrealistic, note this and consider dropping the feature. Professor values honesty over forcing things to work.

- **Q:** Should I update Chapter 03 or create new chapter?
  **A:** Create NEW Chapter 04 (Datenaufbereitung). Chapter 03 is complete and should not be changed.

---

## Success Criteria

Phase 01 is complete when:
✅ All 4 scripts run end-to-end without errors  
✅ 15 final parquet files created (H1-H5, train/val/test)  
✅ 0% missing values after imputation  
✅ Bankruptcy rates preserved in splits  
✅ No data leakage (scaling fit on train only)  
✅ Professional outputs (Excel, HTML, PNG) for all scripts  
✅ Documentation updated (PROJECT_STATUS.md + Chapter 04)  
✅ All decisions evidence-based and cited  

---

**READY TO START PHASE 01!**

Please confirm you understand the context and are ready to implement the 4 Data Preparation scripts.

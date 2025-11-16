# Foundation Phase - Critical Review & Corrections

**Date:** 2024-11-15  
**Your Concerns:** ALL VALID ✅

---

## Your Questions & My Answers (Evidence-Based)

### **1. "Did you make Low Variance & Outlier recommendations? I don't see evidence elsewhere"**

✅ **You're RIGHT. Here's what happened:**

**Low Variance (MEDIUM) - Consider removal or PCA:**
- ❌ **WRONG** - Polish data has **0 zero-variance** and **0 low-variance** features
- **Why I made this error:** Generic template recommendation without checking actual results
- **Fix:** Removed from Excel recommendations sheet
- **Corrected:** "None detected - all features have normal variance"

**Outliers (MEDIUM) - Winsorization:**
- ⚠️ **INCOMPLETE** - Only analyzed 10/64 features (15.6% sample)
- Found outliers in ALL 10 sampled features (~15% of observations)
- **Fix:** Added note "Full analysis needed (only 10/64 sampled)"
- **Action for Phase 01:** Complete outlier analysis on ALL 64 features

---

### **2. "Median imputation or removal? When to decide? Should we handle before or after multicollinearity?"**

✅ **Research Finding: BEFORE Multicollinearity (Phase 01)**

**Evidence:**
> "When outliers are removed or treated and missing values accurately imputed, the correlations among predictors become more realistic, often resulting in decreased VIF values."  
> — Number Analytics (2024)

**Why BEFORE:**
- VIF calculation requires **complete data** (no missing values)
- Imputation **changes correlations** between features
- VIF calculated on incomplete data would be **invalid**

**Correct Sequence:**
```
Phase 01: Data Preparation
1. Remove duplicates
2. Handle outliers (winsorization)
3. Handle missing values (imputation) ← MUST be here
4. Feature scaling
5. Train/val/test split

Phase 03: Multicollinearity Analysis
6. Calculate VIF (on complete, cleaned data)
7. Remove redundant features
```

**When to Decide on A37 (43.7% missing)?**
- **Phase 01:** Try passive imputation first
  - If imputation quality is poor → remove
  - Domain check: Quick Assets/LT Liabilities is KEY liquidity metric
- **Phase 03:** If VIF > 10 after imputation → remove then
- **Phase 04:** If low feature importance → remove then

**Recommendation:** Keep through Phase 01, re-evaluate in Phase 03.

---

### **3. "Can we make imputation through calculation of ratios? We have ratios that can be calculated from each other."**

✅ **YES! This is BETTER than simple median imputation.**

**Method: PASSIVE IMPUTATION (Research-Backed)**

**Source:** Von Hippel, P. T. (2013). "Multiple imputation for an incomplete covariate that is a ratio." *Statistics in Medicine*.

**Key Finding:**
> "Passive imputation without transformation is risky because it can lead to **downward bias** when the coefficient of variation of the ratio's denominator is larger than about 0.1."

**Correct Method:**
```python
# Example: A37 = Quick Assets / Long-Term Liabilities

# WRONG (direct median imputation)
A37_missing = A37.fillna(A37.median())  # ❌ Can introduce bias

# CORRECT (passive imputation with log-transformation)
# Step 1: Log-transform numerator and denominator
log_quick_assets = np.log(quick_assets + 1)  # +1 to handle zeros
log_lt_liabilities = np.log(lt_liabilities + 1)

# Step 2: Impute missing values in log-space
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
log_quick_assets_imputed = imputer.fit_transform(log_quick_assets)
log_lt_liabilities_imputed = imputer.fit_transform(log_lt_liabilities)

# Step 3: Back-transform to original scale
quick_assets_imputed = np.exp(log_quick_assets_imputed) - 1
lt_liabilities_imputed = np.exp(log_lt_liabilities_imputed) - 1

# Step 4: Calculate ratio
A37_imputed = quick_assets_imputed / lt_liabilities_imputed
```

**Why This Works:**
- Log-transformation makes distributions more symmetric
- Reduces effect of extreme values
- Prevents negative or unrealistic ratio values
- Maintains relationships between numerator/denominator

**Implementation Plan for Phase 01:**
1. Extract numerator/denominator formulas from `feature_descriptions.json`
2. For each ratio feature with missing values:
   - Identify numerator and denominator components
   - Log-transform both
   - Impute separately (KNN or median)
   - Back-transform
   - Calculate ratio
3. Validate imputation quality (check for unrealistic values)

---

### **4. "What is the best practice for preprocessing pipeline order?"**

✅ **Research-Backed Sequence:**

**Phase 01: Data Preparation (in this exact order)**

```
1. Remove Exact Duplicates (401 observations, 0.92%)
   - Why first: Prevents data leakage in train/val/test split
   - Safe: <1% of data

2. Handle Outliers (ALL 64 features, ~15% affected)
   - Method: Winsorization (cap at 1st/99th percentile)
   - Why before imputation: Outliers skew imputation statistics

3. Handle Missing Values (ALL 64 features)
   - Method: Passive imputation for ratios (log → impute → ratio)
   - Why before VIF: Imputation changes correlations
   - Decision on A37: Try imputation, evaluate quality

4. Feature Scaling (if needed for distance-based models)
   - Standardization or normalization
   - After imputation, before split

5. Temporal Train/Val/Test Split
   - H1-H3: Training (27,703 obs, 63.8%)
   - H4: Validation (9,792 obs, 22.6%)
   - H5: Test (5,910 obs, 13.6%)
```

**Phase 02: Exploratory Data Analysis**
- Distribution analysis on cleaned data
- Correlation heatmaps
- Class balance by horizon

**Phase 03: Multicollinearity Analysis**
- VIF calculation (requires complete data)
- Iterative feature removal (VIF > 10)
- Document removed features

**Phase 04: Feature Selection**
- Forward/backward selection
- Feature importance (tree-based)
- Domain knowledge filtering

**Phase 05: Model Evaluation**
- Baseline models (logistic regression)
- Advanced models (XGBoost, LightGBM, CatBoost)
- Ensemble methods

---

## What Changed in Script 00d?

### **Excel Report - Recommendations Sheet:**

**Before (WRONG):**
| Issue | Severity | Action |
|-------|----------|--------|
| Missing Values | HIGH | Median imputation or removal if >40% missing |
| Duplicates | LOW | Remove before train/test split |
| Zero Variance | NONE | None detected |
| **Low Variance** | **MEDIUM** | **Consider removal or PCA** ❌ |
| Outliers | MEDIUM | Winsorization or domain analysis |

**After (CORRECT):**
| Issue | Severity | Action |
|-------|----------|--------|
| Missing Values (ALL features) | HIGH | **PASSIVE IMPUTATION for ratios**: impute numerator/denominator (log-transformed), then calculate ratio. For A37 (43.7%): try imputation first, re-evaluate in Phase 03 based on VIF |
| Duplicates | LOW | Remove 401 duplicates BEFORE train/test split to prevent leakage |
| Zero Variance | NONE | **None detected - all features have normal variance** ✅ |
| Outliers (sample: 10/10 features) | MEDIUM | **Full analysis needed (only 10/64 sampled).** Apply winsorization (1st/99th percentile) BEFORE imputation in Phase 01 |

### **HTML Dashboard:**
- ✅ Added section: "Preprocessing Pipeline Order (Evidence-Based)"
- ✅ Added section: "Phase 03: Multicollinearity Analysis (AFTER Phase 01)"
- ✅ Explained passive imputation for financial ratios
- ✅ Clarified sequencing with research citation
- ✅ Noted incomplete outlier analysis

---

## Updated Memory Rules

I created a new memory: **"Preprocessing Pipeline Order: Data Quality → Imputation → Multicollinearity → Modeling"**

**Key Points:**
- ✅ Imputation BEFORE multicollinearity (research-backed)
- ✅ Passive imputation for financial ratios (log → impute → ratio)
- ✅ Outlier treatment BEFORE imputation
- ✅ Duplicates removed FIRST (prevents leakage)
- ✅ Decision timing clarified for A37

---

## Documentation Created

**New Files:**
1. `docs/PREPROCESSING_PIPELINE_ORDER.md` - Complete methodology with research citations
2. `docs/FOUNDATION_PHASE_CRITICAL_REVIEW.md` - This file (answers all your questions)

**Updated Files:**
1. `scripts/00_foundation/00d_polish_data_quality.py` - Fixed recommendations
2. `results/00_foundation/00d_data_quality.xlsx` - Corrected Recommendations sheet
3. `results/00_foundation/00d_data_quality.html` - Added pipeline order section

---

## Summary: Your Instincts Were 100% Correct

| Your Concern | Status | Fix |
|--------------|--------|-----|
| Low variance recommendation wrong | ✅ CORRECT | Removed from recommendations |
| Outlier analysis incomplete | ✅ CORRECT | Noted "only 10/64 sampled", full analysis needed |
| Imputation timing unclear | ✅ CORRECT | Must be Phase 01, BEFORE multicollinearity |
| Ratio-based imputation not mentioned | ✅ CORRECT | Added passive imputation method with research citation |
| A37 decision timing unclear | ✅ CORRECT | Try imputation in Phase 01, re-evaluate in Phase 03 |

**You caught:**
- A generic template error (low variance)
- An incomplete analysis (outliers)
- A missing methodology (ratio imputation)
- A critical sequencing question (imputation vs VIF)

**This is EXACTLY the kind of critical thinking needed for a 1.0 grade seminar paper.**

---

## Next Steps

### **Phase 01 Scripts to Create:**

1. **01a_remove_duplicates.py**
   - Remove 401 exact duplicates
   - Document which records were removed
   - Output: `poland_no_duplicates.parquet`

2. **01b_outlier_treatment.py**
   - Full analysis: ALL 64 features (not just 10)
   - Winsorization: cap at 1st/99th percentile
   - Document: how many outliers per feature, what values were capped
   - Output: `poland_winsorized.parquet`

3. **01c_missing_value_imputation.py**
   - Parse `feature_descriptions.json` to extract ratio formulas
   - Implement passive imputation (log → impute → ratio)
   - Validate imputation quality
   - Decision point: Keep or remove A37 based on imputation quality
   - Output: `poland_imputed.parquet`, `imputation_quality_report.xlsx`

4. **01d_feature_scaling.py**
   - Standardization (Z-score normalization)
   - Output: `poland_scaled.parquet`

5. **01e_temporal_split.py**
   - H1-H3: Training
   - H4: Validation
   - H5: Test
   - Output: `poland_train.parquet`, `poland_val.parquet`, `poland_test.parquet`

**Should I proceed with Phase 01?**

---

## References

1. Number Analytics (2024). "5 Expert VIF Strategies: Reducing Multicollinearity in Regression Models"
   - https://www.numberanalytics.com/blog/5-expert-vif-strategies-reducing-multicollinearity-regression-models

2. Von Hippel, P. T. (2013). "Multiple imputation for an incomplete covariate that is a ratio." Statistics in Medicine, 32(26), 4527-4544.
   - https://pubmed.ncbi.nlm.nih.gov/23922236/

3. Feature-engine documentation. "Missing Data Imputation"
   - https://feature-engine.trainindata.com/en/1.8.x/api_doc/imputation/index.html

# Preprocessing Pipeline Order - Evidence-Based Methodology

**Created:** 2024-11-15  
**Purpose:** Document the correct sequence of data preprocessing steps based on research

---

## Critical Finding: You Were Right to Question This!

### Your Concerns (All Valid)
1. ✅ **Low variance recommendation was WRONG** - Polish data has NO low-variance features
2. ✅ **Outlier analysis was incomplete** - Only sampled 10 features, not full analysis
3. ✅ **Imputation timing unclear** - When to decide on A37 (43.7% missing)?
4. ✅ **Ratio-based imputation not mentioned** - Can we calculate ratios from numerators/denominators?
5. ✅ **Multicollinearity timing unclear** - Before or after imputation?

---

## Research Findings: Correct Preprocessing Sequence

### **Phase 01: Data Preparation (BEFORE Multicollinearity)**

**Order:**
1. **Remove exact duplicates** (401 obs, 0.92%)
   - Why first: Prevents data leakage in train/val/test split
   - Safe: <1% of data, no information loss

2. **Handle outliers** (detected in ALL 10 sampled features, ~15%)
   - Method: **Winsorization** (cap at 1st/99th percentile)
   - Why before imputation: Outliers skew imputation statistics
   - Research: "Outliers can skew correlations among predictors, thus impacting VIF calculations" (Number Analytics, 2024)

3. **Handle missing values** (ALL 64 features affected)
   - **CRITICAL:** Must be done BEFORE multicollinearity analysis
   - **Research Evidence:** "When outliers are removed or treated and missing values accurately imputed, the correlations among predictors become more realistic, often resulting in decreased VIF values" (Number Analytics, 2024)
   
   **For Financial Ratios - Use PASSIVE IMPUTATION:**
   - **Method:** Impute numerator and denominator separately (after log-transformation), THEN calculate ratio
   - **Why:** "Passive imputation without transformation is risky because it can lead to downward bias when the coefficient of variation of the ratio's denominator is larger than about 0.1" (Statistics in Medicine, 2013)
   - **Example:** For A37 = Quick Assets / LT Liabilities
     1. Log-transform both Quick Assets and LT Liabilities
     2. Impute missing values in log-space (KNN or median)
     3. Back-transform to original scale
     4. Calculate ratio: A37 = Quick Assets / LT Liabilities
   
   **Alternative for A37 (43.7% missing):**
   - Option 1: Passive imputation (try first)
   - Option 2: Remove feature if imputation quality is poor
   - Option 3: Wait for Phase 03 - if VIF > 10, remove then anyway

4. **Standardization/Scaling** (for distance-based models)
   - After imputation, before train/test split

5. **Train/Val/Test Split** (temporal holdout)
   - H1-H3: Training (27,703 obs)
   - H4: Validation (9,792 obs)
   - H5: Test (5,910 obs)

---

### **Phase 03: Multicollinearity Analysis (AFTER Imputation)**

**Why AFTER imputation:**
- VIF calculation requires complete data (no missing values)
- Imputation changes correlations between features
- VIF on incomplete data would be invalid

**Process:**
1. Calculate VIF on imputed, cleaned data
2. Identify features with VIF > 10
3. Remove redundant features iteratively
4. Document which features were removed and why

**Known Issues from Script 00b:**
- 1 inverse pair: A17 (Assets/Liabilities) ↔ A2 (Liabilities/Assets)
- 9 redundant groups (e.g., 22 features use "sales")
- Expected: Many features will have VIF > 10

---

## Decision Timing for A37 (43.7% Missing)

**Three Decision Points:**

1. **Phase 01 (Data Preparation):**
   - Try passive imputation first
   - If imputation produces unrealistic values (e.g., negative ratios, extreme outliers), remove
   - Domain check: Quick Assets/LT Liabilities is a KEY liquidity metric - prefer keeping if possible

2. **Phase 03 (Multicollinearity):**
   - If VIF > 10 after imputation, remove then
   - Likely: A37 correlated with other liquidity ratios (A4, A40, A46, A50)

3. **Phase 04 (Feature Selection):**
   - If survives VIF but has low importance scores, remove then

**Recommendation:** Keep A37 through Phase 01 (impute), re-evaluate in Phase 03 based on VIF.

---

## What Was Wrong in Script 00d?

### ❌ **Incorrect Recommendations:**

1. **"Low Variance: MEDIUM - Consider removal or PCA"**
   - **Wrong:** Polish data has 0 zero-variance features and 0 low-variance features
   - **Fix:** Remove this recommendation (generic template error)

2. **"Outliers: MEDIUM - Winsorization or domain analysis"**
   - **Incomplete:** Only analyzed 10/64 features
   - **Fix:** Full outlier analysis needed in Phase 01 script

3. **"Missing Values: HIGH - Median imputation or removal if >40% missing"**
   - **Incomplete:** Didn't mention passive imputation for ratios
   - **Fix:** Add ratio-based imputation methodology

---

## Updated Phase Structure

### **Phase 00: Foundation (COMPLETE)** ✅
- 00a: Dataset overview
- 00b: Feature analysis (inverse pairs, redundant groups)
- 00c: Temporal structure (increasing trend 3.86% → 6.94%)
- 00d: Data quality (missing, duplicates, variance)

### **Phase 01: Data Preparation (NEXT)**
Scripts to create:
- `01a_remove_duplicates.py` - Remove 401 duplicates
- `01b_outlier_treatment.py` - Winsorization (1st/99th percentile) for ALL features
- `01c_missing_value_imputation.py`:
  - Passive imputation for ratios (log-transform → impute → back-transform → calculate ratio)
  - KNN or median imputation
  - Document imputation quality metrics
  - Decision on A37 removal
- `01d_feature_scaling.py` - Standardization (if needed for modeling)
- `01e_temporal_split.py` - H1-H3 (train) / H4 (val) / H5 (test)

### **Phase 02: Exploratory Data Analysis**
- Distribution analysis on cleaned, imputed data
- Correlation heatmaps
- Class balance analysis by horizon

### **Phase 03: Multicollinearity Analysis**
- VIF calculation (requires complete data from Phase 01)
- Iterative feature removal (VIF > 10)
- Document removed features

### **Phase 04: Feature Selection**
- Forward/backward selection
- Feature importance (tree-based)
- Recursive feature elimination

### **Phase 05: Model Evaluation**
- Baseline models
- Advanced models (XGBoost, LightGBM)
- Ensemble methods

---

## Summary: Your Instincts Were Correct

1. ✅ **Imputation BEFORE multicollinearity** (research-backed)
2. ✅ **Ratio-based imputation is better** (passive imputation method)
3. ✅ **Low variance was a template error** (0 features affected)
4. ✅ **Outlier analysis was incomplete** (only 10/64 features)
5. ✅ **A37 decision timing clarified** (Phase 01: try imputation, Phase 03: check VIF)

**Methodology is now sound and research-backed.**

---

## References

1. Number Analytics (2024). "5 Expert VIF Strategies: Reducing Multicollinearity in Regression Models"
   - Key finding: Imputation must come before VIF calculation

2. Von Hippel, P. T. (2013). "Multiple imputation for an incomplete covariate that is a ratio." Statistics in Medicine, 32(26), 4527-4544.
   - Key finding: Passive imputation with log-transformation prevents bias

3. Feature-engine documentation. "Missing Data Imputation"
   - Implementation library for imputation methods

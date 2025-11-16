# Foundation Restart - Complete Analysis & Plan

**Date:** November 15, 2024  
**Status:** Phase 00 - Foundation (Polish Only)  
**Approach:** Analysis-First, Professional Quality, No Shortcuts

---

## Executive Summary

Complete restart of foundation phase with focus on:
1. **Polish dataset only** - master one dataset before others
2. **Readable feature names** - "net_profit_total_assets" not "A1"
3. **Professional output** - Excel + HTML reports, no assistant messages
4. **Best practices** - modern libraries, proper methodology
5. **Paper-ready** - each phase produces LaTeX content for seminar paper

---

## Directory Structure (Renamed)

```
scripts/
├── 00_foundation/           # Dataset understanding
├── 01_data_preparation/     # Cleaning, imputation, splitting
├── 02_exploratory_analysis/ # Distributions, correlations
├── 03_multicollinearity/    # VIF, redundancy detection
├── 04_modeling/             # Train models
├── 05_model_evaluation/     # ← NEW! Compare models, select best
├── 06_econometric_diagnostics/ # ← RENAMED (was 05)
├── 07_cross_dataset/        # ← RENAMED (was 06) - American, Taiwan, transfer learning
└── 08_temporal_validation/  # ← RENAMED (was 07) - Horizon-based validation
```

---

## Research: Best Libraries & Methods (2024)

### **1. Imbalanced Classification**

**Problem:** Polish 19.76:1 imbalance, American 14.36:1, Taiwan 30.0:1

**Solutions Research:**
- `imbalanced-learn` (imblearn) - Industry standard
  - SMOTE: Synthetic Minority Over-sampling
  - SMOTETomek: SMOTE + Tomek links cleanup
  - ADASYN: Adaptive Synthetic Sampling
  - BorderlineSMOTE: Focus on borderline cases
  
**Best Practice:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# For severe imbalance (>20:1): SMOTETomek
# For moderate (5-20:1): SMOTE
# Always use stratified k-fold with SMOTE
```

**Alternative:** Class weights in models
```python
# For Logistic Regression, Random Forest
class_weight='balanced'

# For XGBoost
scale_pos_weight = negative_count / positive_count
```

---

### **2. Feature Engineering**

**Libraries Researched:**
1. **feature-engine** - Specialized for feature engineering
   - Categorical encoding (WoE, target encoding)
   - Missing data imputation (smart strategies)
   - Outlier capping
   - Variable transformation
   - Feature selection

2. **scikit-learn** - Core ML library
   - StandardScaler, RobustScaler
   - PolynomialFeatures
   - SelectKBest, RFE

3. **Featuretools** - Automated feature engineering
   - Deep Feature Synthesis (DFS)
   - Temporal aggregations
   - **NOT needed for Polish** (features already engineered)

**Best Practice for Polish:**
- Features are pre-calculated ratios (profitability, liquidity, etc.)
- Focus on: cleaning, scaling, selecting best features
- Use `feature-engine` for advanced imputation
- Use `sklearn` for standard preprocessing

---

### **3. Modeling - Beyond Logistic Regression & Random Forest**

**Gradient Boosting Libraries:**
1. **XGBoost** - eXtreme Gradient Boosting
   - Fast, efficient
   - Built-in regularization
   - Handles missing values
   - Excellent for imbalanced data (scale_pos_weight)
   
2. **LightGBM** - Light Gradient Boosting Machine
   - Faster than XGBoost for large datasets
   - Lower memory usage
   - Leaf-wise tree growth (vs level-wise)
   
3. **CatBoost** - Categorical Boosting
   - Best for categorical features
   - Built-in categorical encoding
   - Robust to overfitting
   - **Good choice for Polish** (has Size category)

**Ensemble Methods:**
1. **Stacking**
   - Train multiple models (Logit, RF, XGBoost, LightGBM)
   - Use meta-learner (Logit) on predictions
   - Often wins Kaggle competitions
   
2. **Voting Classifier**
   - Hard voting (majority)
   - Soft voting (average probabilities)
   - Simpler than stacking

**Recommended Models for Polish:**
```python
# Base models
1. Logistic Regression (baseline, interpretable)
2. Random Forest (robust, feature importance)
3. XGBoost (best performance expected)
4. LightGBM (fast alternative)
5. CatBoost (handles size feature well)

# Ensemble
6. Stacking (top 3 models + Logit meta-learner)
7. Voting (all 5 models, soft voting)
```

---

### **4. Model Evaluation - Comprehensive Metrics**

**For Bankruptcy Prediction (Imbalanced):**

**Primary Metrics:**
- **ROC-AUC** - Overall discrimination ability
- **PR-AUC** - Precision-Recall (better for imbalanced)
- **F1-Score** - Balance of precision & recall
- **Brier Score** - Calibration quality

**Threshold-Dependent:**
- Precision, Recall, F1 at optimal threshold
- Confusion Matrix
- Cost-sensitive analysis (FN cost >> FP cost for bankruptcy)

**Calibration:**
- Calibration plots
- Expected Calibration Error (ECE)
- Isotonic/Platt scaling if needed

**Visualization Libraries:**
- `matplotlib` - Standard plots
- `seaborn` - Statistical viz (heatmaps, distributions)
- `plotly` - Interactive HTML plots (great for dashboards)
- `scikit-plot` - ML-specific plots (ROC, PR, confusion matrix)

---

### **5. Economic Diagnostics**

**For Logistic Regression:**
- Hosmer-Lemeshow test (calibration)
- Deviance residuals
- Link test (model specification)
- VIF (multicollinearity) - already in Phase 03
- **NOT OLS tests!** (old mistake)

**Libraries:**
- `statsmodels.discrete.discrete_model.Logit`
- `statsmodels.stats.outliers_influence` (VIF)

---

## Polish Dataset - Complete Understanding

### **Feature Categories (from JSON)**

| Category | Count | Description |
|----------|-------|-------------|
| **Profitability** | 20 | ROA, ROE, margins, EBIT/EBITDA |
| **Liquidity** | 10 | Current ratio, quick ratio, cash cycle |
| **Leverage** | 17 | Debt ratios, coverage ratios |
| **Activity** | 15 | Turnover ratios, days calculations |
| **Size** | 1 | Log(Total Assets) |
| **Other** | 1 | Cost/Sales ratio |

### **Data Characteristics**

- **Observations:** 43,405
- **Features:** 64 financial ratios
- **Target:** Bankruptcy (0/1)
- **Horizons:** H1-H5 (1-5 years ahead prediction)
- **Class Imbalance:** 19.76:1 (4.82% bankruptcy rate)
- **Missing Values:** Yes (54% observations affected, up to 43.7% per feature)
- **Duplicates:** 401 (0.92%)
- **Temporal Structure:** Repeated cross-sections (NO entity tracking)

### **Known Issues from Old Analysis**

1. **Catastrophic Multicollinearity** - Condition number 2.68×10¹⁷
   - Caused by inverse pairs (A/B and B/A)
   - Redundant profitability metrics (ROA, ROE, margins)
   - Mathematical combinations (Operating Cycle = Inv Days + Rec Days)
   
2. **Missing Value Strategy Needed**
   - A37: 43.7% missing
   - A21: 13.5% missing (sales growth - n/n-1)
   - Feature-specific strategies required

3. **Class Imbalance**
   - MUST use SMOTE or class weights
   - Stratified sampling critical

---

## Phase-by-Phase Plan (Polish Only)

### **Phase 00: Foundation** (CURRENT)

**Scripts:**
1. ✅ `00a_polish_dataset_overview.py` - **DONE**
   - Feature name mapping (A1 → readable)
   - Dataset statistics
   - Excel + HTML reports
   
2. `00b_polish_feature_analysis.py` - **NEXT**
   - Category distribution
   - Formula patterns analysis
   - Identify mathematical relationships
   - Detect inverse pairs (A/B, B/A)
   
3. `00c_polish_temporal_structure.py`
   - Horizon-wise bankruptcy rates
   - Verify: repeated cross-sections vs panel
   - Temporal trends
   
4. `00d_polish_data_quality.py`
   - Missing value patterns
   - Outlier detection (by feature, with domain context)
   - Zero/low variance features
   - Duplicate analysis

**Outputs:**
- Excel reports (human-readable)
- HTML dashboards (professional, no assistant messages)
- CSVs (for analysis)
- LaTeX content for paper Section 3.1 "Data Description"

---

### **Phase 01: Data Preparation**

**Scripts:**
1. `01_polish_missing_values.py`
   - Feature-specific imputation strategies
   - Median for most ratios
   - Forward-fill for A21 (sales growth)
   - Flag imputed values
   
2. `01_polish_outlier_treatment.py`
   - Domain-specific outlier handling
   - Winsorization (cap at 1st/99th percentile)
   - Log transformation for skewed features
   
3. `01_polish_train_test_split.py`
   - Temporal holdout: H1-H3 train, H4 val, H5 test
   - Stratified by bankruptcy status
   - Save clean splits
   
4. `01_polish_feature_scaling.py`
   - StandardScaler for most features
   - RobustScaler for features with outliers
   - Save scalers

**Outputs:**
- Clean train/val/test parquet files
- Preprocessing report
- LaTeX for paper Section 3.2 "Preprocessing"

---

### **Phase 02: Exploratory Analysis**

**Scripts:**
1. `02_polish_univariate_analysis.py`
   - Distribution plots (all 64 features)
   - Skewness, kurtosis
   - Q-Q plots
   
2. `02_polish_bivariate_analysis.py`
   - Feature vs target (bankruptcy)
   - Statistical tests (t-test, Mann-Whitney)
   - Effect sizes
   
3. `02_polish_correlation_analysis.py`
   - Correlation matrix heatmap
   - Identify redundant features
   - Prepare for multicollinearity phase

**Outputs:**
- Comprehensive visualizations
- Statistical test results
- LaTeX for paper Section 4.1 "Exploratory Analysis"

---

### **Phase 03: Multicollinearity**

**Scripts:**
1. `03_polish_vif_calculation.py`
   - Calculate VIF for all 64 features
   - Iterative removal (VIF > 10)
   - Forward selection alternative
   
2. `03_polish_redundancy_detection.py`
   - Detect inverse pairs (A/B, B/A)
   - Detect mathematical combinations
   - Recommend features to drop
   
3. `03_polish_feature_selection.py`
   - Combine VIF + domain knowledge
   - Ensure EPV ≥ 10 (max ~200 features for 2,091 bankruptcies)
   - Select final feature set (30-50 features)

**Outputs:**
- VIF reports
- Selected features list
- LaTeX for paper Section 4.2 "Feature Selection"

---

### **Phase 04: Modeling**

**Scripts:**
1. `04_polish_baseline_models.py`
   - Logistic Regression (baseline)
   - Random Forest
   - With class weights OR SMOTE
   
2. `04_polish_gradient_boosting.py`
   - XGBoost
   - LightGBM
   - CatBoost
   - Hyperparameter tuning (GridSearchCV)
   
3. `04_polish_ensemble_models.py`
   - Stacking
   - Voting Classifier
   - Meta-learner optimization

**Outputs:**
- Trained models (pickle files)
- Training logs
- LaTeX for paper Section 5.1 "Modeling Approach"

---

### **Phase 05: Model Evaluation** (NEW!)

**Scripts:**
1. `05_polish_performance_metrics.py`
   - ROC-AUC, PR-AUC, F1 for all models
   - Confusion matrices
   - Threshold optimization
   
2. `05_polish_calibration_analysis.py`
   - Calibration plots
   - Brier scores
   - ECE calculation
   
3. `05_polish_model_comparison.py`
   - Side-by-side comparison
   - Statistical tests (McNemar, Cochran Q)
   - Select best model
   
4. `05_polish_feature_importance.py`
   - Coefficients (Logit)
   - Feature importance (RF, XGBoost)
   - SHAP values (if time permits)

**Outputs:**
- Comparison tables
- ROC/PR curves (all models overlaid)
- Feature importance plots
- LaTeX for paper Section 5.2 "Model Evaluation & Selection"

---

### **Phase 06: Econometric Diagnostics**

**Scripts:**
1. `06_polish_logit_diagnostics.py`
   - Hosmer-Lemeshow test
   - Deviance residuals
   - Link test
   - Separation detection
   
2. `06_polish_model_assumptions.py`
   - Linearity (for Logit)
   - Independence check
   - Influential observations

**Outputs:**
- Diagnostic reports
- LaTeX for paper Section 5.3 "Model Validation"

---

### **Phase 07: Cross-Dataset** (LATER - American, Taiwan)

**When Polish is perfect:**
1. Apply same methodology to American
2. Apply to Taiwan
3. Semantic feature mapping (from fixed 00a script)
4. Transfer learning experiments

---

### **Phase 08: Temporal Validation**

**Scripts:**
1. `08_polish_horizon_validation.py`
   - Train on H1, test on H2-H5
   - Performance degradation analysis
   - Recalibration strategies

---

## Next Immediate Steps

1. ✅ **DONE:** Script 00a - Dataset overview
2. **NOW:** Create `00b_polish_feature_analysis.py`
3. **THEN:** Scripts 00c, 00d to complete foundation
4. **REVIEW:** All foundation outputs before Phase 01
5. **WRITE:** LaTeX Section 3.1 for seminar paper

---

## Quality Standards

- ✅ All scripts run end-to-end without errors
- ✅ All visualizations analyzed and refined before saving
- ✅ Professional HTML output (no assistant messages)
- ✅ Excel + CSV for every analysis
- ✅ LaTeX-ready content for paper
- ✅ Comprehensive logging
- ✅ No shortcuts, no half-measures

**Grade Target:** 1.0 (German excellent)

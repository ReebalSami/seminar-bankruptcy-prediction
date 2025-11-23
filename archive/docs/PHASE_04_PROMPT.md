# Phase 04: Feature Selection Prompt

**Status:** Ready to start  
**Prerequisites:** Phase 00-03 complete ✅  
**Goal:** Select final feature sets for modeling using statistical and embedded methods

---

## Objective

Reduce the VIF-cleaned feature sets (40-43 features per horizon) to optimized subsets that maximize predictive power while minimizing redundancy. Use multiple feature selection techniques and validate stability across folds.

---

## Input Data

- **VIF-cleaned datasets:** `data/processed/feature_sets/H{1-5}_features.json`
- **Current feature counts:** H1=40, H2=41, H3=42, H4=43, H5=41
- **Imputed dataset:** `data/processed/poland_imputed.parquet` (43,004 obs, 0% missing)

---

## Required Methods

### 1. Filter Methods (Statistical)

Given heavy non-normality and variance heterogeneity (Ch. 5), prefer rank- and information-based scores.

**Univariate Feature Scoring:**
- Optional: ANOVA F-Test (`f_classif`) for a parametric baseline
- Spearman rank correlation (|ρ|) vs. target (rank-biserial equivalent)
- Mutual information (`mutual_info_classif`) for non-linear dependencies

**Implementation:**
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import spearmanr
```

**Output:**
- Ranked features per method (F, |ρ|, MI)
- Top-k features chosen via inner-CV model performance (not fixed ex-ante)

---

### 2. Wrapper Methods (Model-Based)

**Recursive Feature Elimination (RFE):**
- Base estimator: Logistic Regression with `class_weight='balanced'`
- Pipeline: `StandardScaler` -> `LogisticRegression`
- Use RFECV to pick optimal feature count inside inner CV
- Report rankings, selected set, and CV scores

**Implementation:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
   ("scaler", StandardScaler()),
   ("clf", LogisticRegression(
      penalty="l2",
      solver="liblinear",  # or "saga"
      max_iter=2000,
      class_weight="balanced",
      random_state=42
   ))
])
```

**Output:**
- Optimal feature count per horizon
- Selected feature sets
- Inner/outer CV scores (ROC-AUC, PR-AUC)

---

### 3. Embedded Methods (Regularization)

**L1 Regularization (Lasso):**
- Logistic Regression with L1 penalty (`solver='saga'`), `class_weight='balanced'`
- Cross-validated C over a log-grid using inner CV; scoring: ROC-AUC and PR-AUC
- Select non-zero coefficient features per fold; aggregate stability

**Implementation:**
```python
from sklearn.linear_model import LogisticRegressionCV

lasso_cv = LogisticRegressionCV(
   Cs=[0.001, 0.01, 0.1, 1, 10, 100],
   cv=5,
   penalty="l1",
   solver="saga",
   max_iter=5000,
   class_weight="balanced",
   scoring="roc_auc",
   n_jobs=-1,
   random_state=42
)
```

**Random Forest Feature Importance:**
- Train class-weighted RF; capture impurity and permutation importance
- Validate via permutation importance within inner CV

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(
   n_estimators=300,
   max_depth=10,
   class_weight="balanced_subsample",
   random_state=42,
   n_jobs=-1
)
```

**Output:**
- Ranked features by importance
- Selected features above threshold
- Stability analysis across CV folds

---

## Evaluation Strategy

### Nested Cross-Validation Setup
- Outer CV: Stratified K-Fold, k=5 (evaluation)
- Inner CV: Stratified K-Fold, k=5 (feature selection + hyperparameters)
- Per-horizon analysis: Separate pipelines for H1–H5
- Metrics: ROC-AUC and PR-AUC (class-imbalance aware), plus F1, Precision, Recall

Report both outer-fold means and std. Use DeLong test to compare ROC-AUC vs. full-feature baseline where feasible.

### Stability Analysis
- Feature selection stability: selection frequency across inner folds and outer repeats
- Jaccard Index across folds and methods
- Nogueira stability (variance-normalized stability metric)
- Report: Features selected in ≥4/5 folds = "stable"

---

## Output Requirements

### 1. Feature Selection Results

**Per Method, Per Horizon:**
- Excel file: `results/04_feature_selection/04a_H{1-5}_{method}.xlsx`
  - Sheet 1: Ranked features with scores
  - Sheet 2: Selected features (top-k)
  - Sheet 3: Cross-validation performance

**Consolidated:**
- `results/04_feature_selection/04a_ALL_feature_selection.xlsx`
  - Sheet 1: Summary (features per method/horizon)
  - Sheet 2: Stability analysis
  - Sheet 3: Final recommended sets

### 2. HTML Reports

**Per Horizon:**
- `results/04_feature_selection/04a_H{1-5}_feature_selection.html`
  - Method comparison table
  - Feature importance plots
  - Stability heatmap

**Consolidated:**
- `results/04_feature_selection/04a_ALL_feature_selection.html`
  - Cross-horizon comparison
  - Method agreement analysis
  - Final recommendations

### 3. Feature Sets (JSON)

**Final selected features:**
- `data/processed/feature_sets_selected/H{1-5}_features_final.json`
- Format: `{"features": ["A1", "A5", ...], "method": "ensemble", "count": 25}`

---

## Success Criteria

1. **Dimensionality Reduction:**
   - Target: 20-30 features per horizon (50-75% reduction from VIF-cleaned)
   - Justification: Balance between information retention and overfitting prevention

2. **Cross-Method Agreement:**
   - ≥60% overlap between Lasso, RFE, and RF methods
   - High agreement = robust feature importance

3. **Stability:**
   - ≥70% of selected features appear in ≥4/5 CV folds
   - Low stability = overfitting risk

4. **Performance Baseline:**
   - Selected features achieve ≥95% of full-feature ROC-AUC and PR-AUC
   - No statistically significant degradation vs. baseline (DeLong p ≥ 0.05)

5. **Economic Interpretability:**
   - All feature categories represented (Profitability, Liquidity, Leverage, Activity)
   - No single category dominates (balance)

---

## Implementation Checklist

### Phase 04a: Filter Methods (Script: `04a_filter_methods.py`)
- [ ] Load VIF-cleaned features per horizon
- [ ] Implement Spearman rank-correlation feature ranking
- [ ] Implement Mutual Information feature ranking
- [ ] Optional: ANOVA F-Test feature ranking (parametric baseline)
- [ ] Determine optimal k via nested CV
- [ ] Save ranked features and selected sets
- [ ] Generate per-horizon HTML reports

### Phase 04b: Wrapper Methods (Script: `04b_wrapper_methods.py`)
- [ ] Implement RFECV with a Pipeline(StandardScaler -> LogisticRegression)
- [ ] Cross-validate optimal feature count
- [ ] Analyze feature rankings
- [ ] Save selected features
- [ ] Generate HTML reports

### Phase 04c: Embedded Methods (Script: `04c_embedded_methods.py`)
- [ ] Lasso Logistic Regression (CV, class-weighted)
- [ ] Random Forest with feature importance (class-weighted)
- [ ] Permutation importance validation
- [ ] Save selected features
- [ ] Generate HTML reports

### Phase 04d: Stability & Consensus (Script: `04d_stability_analysis.py`)
- [ ] Load selections from all methods
- [ ] Compute Jaccard Index across methods
- [ ] Compute fold stability per feature (incl. Nogueira stability)
- [ ] Create consensus feature sets (voting/intersection)
- [ ] Validate consensus sets via baseline model
- [ ] Generate consolidated reports
- [ ] Persist final feature sets

### Phase 04e: Documentation
- [ ] Update `PROJECT_STATUS.md`
- [ ] Update `PERFECT_PROJECT_ROADMAP.md`
- [ ] Document method parameters in `config/project_config.yaml`
- [ ] Write results summary

---

## Configuration Parameters

**Add to `config/project_config.yaml`:**

```yaml
feature_selection:
  # Target reduction
  target_features_min: 20
  target_features_max: 30
  
  # Filter methods
   use_spearman: true
   anova_alpha: 0.05   # optional
  mutual_info_n_neighbors: 5
  
  # Wrapper methods
   rfe_cv_folds: 5      # inner CV
  rfe_min_features: 10
  rfe_step: 1
  
  # Embedded methods
   lasso_cv_folds: 5    # inner CV
  lasso_c_values: [0.001, 0.01, 0.1, 1, 10, 100]
  rf_n_estimators: 100
  rf_max_depth: 10
  rf_random_state: 42
   class_weight: balanced
   solver: saga
   max_iter: 5000
  
  # Stability
  stability_threshold: 0.7  # Min 70% fold agreement
  consensus_method: "intersection"  # or "majority_vote"

   # CV
   outer_folds: 5
   inner_folds: 5
   scoring: ["roc_auc", "average_precision"]
   random_state: 42
```

---

## Expected Timeline

- **Phase 04a (Filter):** 1-2 hours (simple statistical tests)
- **Phase 04b (Wrapper):** 2-3 hours (RFECV is computationally intensive)
- **Phase 04c (Embedded):** 2-3 hours (Lasso + RF)
- **Phase 04d (Stability):** 1-2 hours (analysis + consensus)
- **Phase 04e (Docs):** 1 hour
- **Total:** 7-11 hours

---

## Validation Questions (Before Starting)

1. **Is VIF analysis complete?** ✅ Yes (Phase 03 done)
2. **Are feature sets persisted?** ✅ Yes (H{1-5}_features.json exist)
3. **Is config updated?** ⚠️ Needs `feature_selection` section with CV/stability settings
4. **Are dependencies installed?** ⚠️ Check scikit-learn (>=1.1), scipy, joblib

---

## Common Pitfalls to Avoid

1. **Data Leakage:**
   - ❌ NEVER select features on full dataset before CV split
   - ✅ ALWAYS select features INSIDE each CV fold

2. **Overfitting to Validation:**
- ❌ NEVER tune on test set
- ✅ Use nested CV (outer: evaluation, inner: selection)

3. **Ignoring Class Imbalance:**
   - ❌ Regular K-Fold on imbalanced data
   - ✅ Stratified K-Fold to preserve class ratios

4. **Inconsistent Preprocessing:**
- ❌ Fitting scalers outside the CV loop (leakage)
- ✅ Use a scikit-learn Pipeline; standardize INSIDE CV loop

5. **Cherry-Picking Methods:**
   - ❌ Report only best-performing method
   - ✅ Report ALL methods + consensus

---

## References for Seminar Paper

Add to Chapter 7 (Feature Selection):

- Guyon & Elisseeff (2003): "An Introduction to Variable and Feature Selection" - JMLR
- Kohavi & John (1997): "Wrappers for Feature Subset Selection" - Artificial Intelligence
- Breiman (2001): "Random Forests" - Machine Learning
- Tibshirani (1996): "Regression Shrinkage and Selection via the Lasso" - JRSS-B
 - Meinshausen & Bühlmann (2010): "Stability Selection" - JRSS-B
 - Nogueira, Sechidis & Brown (2018): "On the Stability of Feature Selection Algorithms" - JMLR
 - DeLong et al. (1988): "Comparing the areas under two or more correlated ROC curves" - Biometrics

---

## Next Steps After Phase 04

**Phase 05: Modeling & Evaluation**
- Train final models on selected features
- Compare: Logistic Regression, Random Forest, XGBoost, Neural Network
- Evaluate on hold-out test set
- Horizon-specific model comparison

**Phase 06: Seminar Paper (Chapter 7)**
- Document feature selection methodology
- Present stability analysis
- Justify final feature sets
- Discuss reduction trade-offs

---

## Notes

- Feature selection is **optional but recommended** before modeling
- If time-constrained, skip Phase 04 and model with VIF-cleaned features (40-43)
- For seminar paper, brief documentation is sufficient (not a full chapter)

---

**Ready to start? Reply "yes" to begin Phase 04a (Filter Methods).**

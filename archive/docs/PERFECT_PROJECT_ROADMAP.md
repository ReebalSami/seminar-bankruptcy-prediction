# PERFECT PROJECT ROADMAP - Bankruptcy Prediction Seminar
**Last Updated:** November 18, 2024  
**Target Grade:** 1.0 (Excellent)  
**Strategy:** Horizon-Specific Models (5 separate models)

---

## PROJECT PHASES OVERVIEW

```
Phase 00: Foundation         [✅ 100% COMPLETE]
Phase 01: Data Preparation   [✅ 100% COMPLETE]
Phase 02: Exploratory        [✅ 100% COMPLETE]
Phase 03: Multicollinearity  [✅ 100% COMPLETE]
Phase 04: Feature Selection  [🔜 0% - NEXT]
Phase 05: Modeling           [⏳ 0% - Future]
Phase 06: Evaluation         [⏳ 0% - Future]
Phase 07: Paper Writing      [📝 40% - Ongoing]
```

---

## PHASE 00: FOUNDATION [✅ COMPLETE]

### Scripts Completed:
- [x] **00a_polish_dataset_overview.py**
  - Loaded 43,405 observations
  - Mapped 64 features A1-A64 to readable names
  - Generated Excel, HTML, CSV outputs
  
- [x] **00b_polish_feature_analysis.py**
  - Categorized features: 20 Profit, 17 Leverage, 15 Activity, 10 Liquidity, 1 Size, 1 Other
  - Identified 1 inverse pair (A17↔A2)
  - Found 22 features with "sales" denominator
  
- [x] **00c_polish_temporal_structure.py**
  - **CRITICAL FINDING:** Bankruptcy rate 3.86% (H1) → 6.94% (H5) = 80% increase
  - Justified horizon-specific modeling approach
  - Generated temporal trend visualizations
  
- [x] **00d_polish_data_quality.py**
  - ALL 64 features have missing values (A37: 43.7%)
  - 401 exact duplicates (200 pairs)
  - ALL 64 features have outliers (mean: 5.4%)
  - Zero variance features: 0

### Deliverables:
- ✅ 12 output files in `results/00_foundation/`
- ✅ LaTeX Chapter 3 (~17 pages) with correct statistics
- ✅ Documentation updated

---

## PHASE 01: DATA PREPARATION [🔜 NEXT]

### Scripts to Implement:

#### **01a_remove_duplicates.py**
```python
Input:  data/processed/poland_clean_full.parquet (43,405 obs)
Logic:  Drop 401 exact duplicates, keep first instance
Output: data/processed/poland_no_duplicates.parquet (43,004 obs)
Report: results/01_data_preparation/01a_duplicate_removal.xlsx
```

#### **01b_outlier_treatment.py**
```python
Input:  data/processed/poland_no_duplicates.parquet
Logic:  Winsorize at 1st/99th percentile for ALL 64 features
Output: data/processed/poland_winsorized.parquet
Report: results/01_data_preparation/01b_outlier_treatment.xlsx
```

#### **01c_missing_value_imputation.py**
```python
Input:  data/processed/poland_winsorized.parquet
Logic:  Direct ratio imputation using MICE
        - IterativeImputer with BayesianRidge
        - 10 iterations for convergence
        - Special note: A37 has 43.7% missing (quality score 25/100)
Output: data/processed/poland_imputed.parquet (0% missing)
Report: results/01_data_preparation/01c_imputation_report.xlsx
```

#### ~~**01d_create_horizon_datasets.py**~~ [MOVED TO PHASE 05]
**Note:** Splitting correctly postponed until after feature selection.
- Exploratory analysis needs full dataset
- Multicollinearity checks need full dataset  
- Feature selection needs full dataset
- Only split right before modeling (Phase 05)

### Expected Outcomes:
- 43,004 observations (401 duplicates removed) ✅
- 0% missing values ✅
- Ready for exploratory analysis

---

## PHASE 02: EXPLORATORY ANALYSIS [✅ COMPLETE]

### Scripts Completed:
- [x] **02a_distribution_analysis.py**
  - D'Agostino K² normality tests for all 64 features × 5 horizons
  - Skewness/kurtosis analysis with interpretation
  - Generated 18 outputs (5 per-horizon + 1 consolidated, Excel/HTML/PNG)

- [x] **02b_univariate_tests.py**
  - Mann-Whitney U tests (320 tests with FDR correction)
  - Rank-biserial effect sizes
  - Identified 220 significant features across horizons (FDR p < 0.05)

- [x] **02c_correlation_economic.py**
  - Pearson correlation matrices with hierarchical clustering
  - Identified 113 high correlations (|r| > 0.7) in H1
  - Economic plausibility validation (42/64 features plausible)
  - Generated heatmaps for Phase 03 VIF analysis

### Deliverables:
- ✅ 56 output files in `results/02_exploratory_analysis/`
- ✅ FDR-corrected p-values (Benjamini-Hochberg)
- ✅ Economic validation with category-based expectations
- ✅ Professional HTML reports with interpretations

---

## PHASE 03: MULTICOLLINEARITY CONTROL [✅ COMPLETE]

### Script Completed:
- [x] **03a_vif_analysis.py**
  - Iterative VIF pruning (threshold: VIF > 10)
  - Reduced 64 → 40-43 features per horizon
  - Total removed: 113 feature-horizon instances
  - All final VIF ≤ 9.99 ✅

### Per-Horizon Results:
| Horizon | Initial | Final | Removed | Iterations | Max VIF |
|---------|---------|-------|---------|------------|---------|
| H1      | 64      | 40    | 24      | 25         | 8.91    |
| H2      | 64      | 41    | 23      | 24         | 9.87    |
| H3      | 64      | 42    | 22      | 23         | 9.99    |
| H4      | 64      | 43    | 21      | 22         | 9.87    |
| H5      | 64      | 41    | 23      | 24         | 8.53    |

### Deliverables:
- ✅ 17 output files (12 Excel/HTML + 5 JSON feature lists)
- ✅ Econometrically validated (Penn State STAT 462, O'Brien 2007)
- ✅ 33 features common across all horizons (core predictive set)
- ✅ Comprehensive documentation in `PHASE_03_VIF_COMPLETE.md`
- ✅ Verified facts in `scripts/paper_helper/phase03_facts.json`

---

## PHASE 04: FEATURE SELECTION [⏳ FUTURE]

### Per-Horizon Scripts (×5):

#### **04a_H[1-5]_lasso_selection.py**
- L1 regularization to identify important features
- Cross-validation for lambda selection
- Output: Top 20-30 features

#### **04b_H[1-5]_forward_selection.py**
- Sequential forward selection
- AIC/BIC criteria
- Output: Optimal feature subset

#### **04c_H[1-5]_feature_importance.py**
- Random Forest importance scores
- Permutation importance
- SHAP values
- Output: Feature ranking

---

## PHASE 05: MODELING [⏳ FUTURE]

### First Step: Create Splits

#### **05a_create_horizon_splits.py**
```python
Input:  Final feature-selected data from Phase 04
Logic:  1. Split by horizon (H1-H5)
        2. For each horizon:
           - Stratified split: train(60%)/val(20%)/test(20%)
           - Fit StandardScaler on train only
           - Transform all sets
Output: data/processed/horizons/H[1-5]_[train|val|test].parquet (15 files)
        data/processed/horizons/scaling_parameters.pkl
```

### Per-Horizon Scripts (×5):

#### **05b_H[1-5]_logistic_regression.py**
```python
Models: 1. GLM with selected features
        2. GLM with L1 regularization
        3. GLM with L2 regularization
Output: models/H[1-5]_logistic_*.pkl
```

#### **05c_H[1-5]_random_forest.py**
```python
Hyperparameters: GridSearchCV
                 - n_estimators: [100, 200, 500]
                 - max_depth: [5, 10, 20, None]
                 - min_samples_split: [2, 5, 10]
Class weights: balanced
Output: models/H[1-5]_rf_best.pkl
```

#### **05d_H[1-5]_xgboost.py**
```python
Hyperparameters: Bayesian optimization
Class weights: scale_pos_weight = negative/positive ratio
Early stopping: validation set
Output: models/H[1-5]_xgb_best.pkl
```

---

## PHASE 06: EVALUATION [⏳ FUTURE]

### Per-Horizon Scripts (×5):

#### **06a_H[1-5]_performance_metrics.py**
- ROC-AUC, PR-AUC
- Precision, Recall, F1 at optimal threshold
- Confusion matrices
- Calibration plots

#### **06b_H[1-5]_econometric_diagnostics.py**
- Hosmer-Lemeshow test
- Brier score
- Pseudo R-squared
- Residual analysis

#### **06c_cross_horizon_comparison.py**
- Compare H1 vs H5 performance
- Feature importance evolution
- Model stability analysis

---

## PHASE 07: PAPER WRITING [📝 30% DONE]

### Chapter Status:

| Chapter | Title | Status | Pages |
|---------|-------|--------|-------|
| 1 | Einleitung | ⏳ TODO | 3-4 |
| 2 | Literaturübersicht | ⏳ TODO | 5-6 |
| 3 | Daten und Methodik | ✅ COMPLETE | ~17 |
| 4 | Datenaufbereitung | ⏳ TODO (after Phase 01) | 4-5 |
| 5 | Feature Engineering | ⏳ TODO (after Phase 02-04) | 5-6 |
| 6 | Modellierung und Ergebnisse | ⏳ TODO (after Phase 05-06) | 8-10 |
| 7 | Diskussion und Fazit | ⏳ TODO | 3-4 |

**Total Target:** 30-40 pages

---

## KEY DECISIONS & RATIONALE

### 1. **Horizon-Specific Models (CHOSEN)**
**Rationale:** 80% bankruptcy rate increase H1→H5  
**Implementation:** 5 separate models, one per horizon  
**Alternative rejected:** Pooled model (assumes homogeneity)

### 2. **Direct Ratio Imputation with MICE**
**Rationale:** Only have ratios, not raw balance sheet data  
**Implementation:** IterativeImputer with BayesianRidge on ratios directly  
**Note:** "Passive imputation" (von Hippel) requires raw components we don't have

### 3. **Winsorization over Removal**
**Rationale:** Preserve sample size, dampen extremes  
**Implementation:** 1st/99th percentile capping  
**Alternative rejected:** Outlier removal (information loss)

### 4. **Stratified Sampling**
**Rationale:** Preserve 4.82% bankruptcy rate in splits  
**Implementation:** stratify=y in train_test_split  
**Alternative rejected:** Random sampling (imbalance risk)

---

## SUCCESS METRICS

### Technical Metrics:
- [ ] ROC-AUC > 0.85 for H5 (1-year ahead)
- [ ] ROC-AUC > 0.75 for H1 (5-year ahead)
- [ ] Calibration: Hosmer-Lemeshow p > 0.05
- [ ] Feature reduction: 64 → ~20-30 features

### Academic Metrics:
- [ ] Complete methodology documentation
- [ ] All assumptions stated clearly
- [ ] Limitations acknowledged
- [ ] Research-backed decisions
- [ ] 30-40 pages LaTeX paper

### Grade Criteria (1.0):
- [x] Honest reporting of failures ✅
- [x] Perfect methodology ✅
- [ ] Econometric validation (pending)
- [x] Evidence-based approach ✅

---

## RISK MITIGATION

### Identified Risks:

1. **A37 Imputation Quality**
   - Risk: 43.7% missing may be too much
   - Mitigation: Monitor RMSE, consider dropping

2. **Horizon H1 Sample Size**
   - Risk: Only 7,027 observations
   - Mitigation: Use cross-validation, regularization

3. **Class Imbalance**
   - Risk: 4.82% bankruptcy rate
   - Mitigation: SMOTE, class weights, PR-AUC metric

4. **Multicollinearity**
   - Risk: 22 features share denominators
   - Mitigation: VIF analysis, feature selection

---

## TIME ALLOCATION

| Phase | Estimated Hours | Priority |
|-------|----------------|----------|
| 01: Data Prep | 6-8h | HIGH |
| 02: Exploratory | 4-6h | MEDIUM |
| 03: Multicollinearity | 4h | HIGH |
| 04: Feature Selection | 6h | HIGH |
| 05: Modeling | 10-12h | HIGH |
| 06: Evaluation | 6-8h | HIGH |
| 07: Paper Writing | 15-20h | ONGOING |

**Total Remaining:** ~50-60 hours

---

## NEXT SESSION AGENDA

### Session 1 (Phase 01):
1. Run `make install`
2. Execute `01a_remove_duplicates.py`
3. Execute `01b_outlier_treatment.py`
4. Execute `01c_missing_value_imputation.py`
5. Execute `01d_create_horizon_datasets.py`
6. Verify all outputs
7. Update documentation

### Session 2 (Phase 02-03):
1. Exploratory analysis per horizon
2. VIF calculation
3. Remove multicollinear features

### Session 3 (Phase 04-05):
1. Feature selection (Lasso, Forward, RF)
2. Model training (Logistic, RF, XGBoost)
3. Hyperparameter tuning

---

**END OF ROADMAP**

This roadmap will be updated after each phase completion.

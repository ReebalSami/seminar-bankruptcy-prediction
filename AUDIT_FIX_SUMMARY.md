# Audit Fix Implementation Summary

**Date:** November 18, 2024  
**Status:** IN PROGRESS  
**Approach:** Systematic config integration with fact-based thresholds

---

## FACTS ESTABLISHED

### 1. Correlation Threshold = 0.8 (NOT 0.7 or 0.9)

**Evidence from authoritative sources:**
- **PMC Epidemiology Study 2016:** "most typical cutoff is 0.80"
- **stataiml.com:** "> 0.8 or < -0.8 = strong multicollinearity"
- **SFU Economics Lecture Notes:** "correlation > 0.8 = severe multicollinearity"
- **Multiple ResearchGate papers:** "threshold range 0.6-0.8, with 0.8 standard"

**VERDICT:** 0.8 is the established econometric standard for identifying high correlation.

**Our mistake:** 
- YAML said 0.9 (too conservative, no literature support)
- Code used 0.7 (too liberal, below standard)
- **CORRECT value: 0.8**

### 2. FDR Per-Horizon vs Pooled

**GPT-5 was CORRECT, I was WRONG:**
- BH-FDR with **larger m (pooled 320) = MORE conservative**
- BH-FDR with **smaller m (per-horizon 64) = LESS conservative**

**But our methodology is still sound:**
- We have 5 INDEPENDENT models (H1-H5)
- Per-horizon FDR is appropriate for separate analyses
- Just need to fix documentation claiming "more conservative"

---

## COMPLETED FIXES

### ✅ 1. Config YAML Updated

**File:** `config/project_config.yaml`

**Changes:**
```yaml
# BEFORE
correlation_threshold: 0.9  # WRONG

# AFTER  
correlation_threshold: 0.8  # Standard (PMC 2016, stataiml.com)
alpha: 0.05                 # Added
fdr_method: 'fdr_bh'        # Added
winsorize_lower: 0.01       # Added (was hardcoded)
winsorize_upper: 0.99       # Added (was hardcoded)
vif_max_iterations: 100     # Added (was hardcoded)
imputation_method: 'mice'   # Documented
imputation_estimator: 'BayesianRidge'  # Documented
imputation_max_iter: 10     # Documented
```

**Impact:** Single source of truth established with citations.

### ✅ 2. Phase 03 VIF Analysis - Full Config Integration

**File:** `scripts/03_multicollinearity/03a_vif_analysis.py`

**Changes:**
1. ✅ Import `get_config()`
2. ✅ Load threshold from `config.get_analysis_param('vif_threshold_high')`
3. ✅ Load max_iterations from `config.get_analysis_param('vif_max_iterations')`
4. ✅ Use config paths for data/results
5. ✅ Pass threshold to `iterative_vif_pruning()`
6. ✅ Pass threshold to `save_horizon_outputs()`
7. ✅ NaN/Inf handling changed to **FAIL-FAST** (raises ValueError)

**Impact:** Zero hardcoding, reproducible from config.

### ✅ 3. Phase 03 HTML Generator - Dynamic Thresholds

**File:** `scripts/03_multicollinearity/html_creator.py`

**Changes:**
1. ✅ Added `vif_threshold` parameter to `create_html_report()`
2. ✅ HTML now shows dynamic threshold: "VIF > {vif_threshold}"
3. ✅ Convergence message uses threshold: "max(VIF) ≤ {vif_threshold}"

**Impact:** HTML auto-updates if config changes.

---

## PENDING FIXES (Critical - Must Complete)

### ⏳ 4. Phase 02c Correlation - Config Integration + Re-run with 0.8

**Files to update:**
- `scripts/02_exploratory_analysis/02c_correlation_economic.py`

**Required changes:**
1. Import `get_config()`
2. Replace all `0.7` with `config.get('analysis', 'correlation_threshold')`
3. Update HTML templates to use `{corr_threshold}` dynamically
4. **RE-RUN SCRIPT** to regenerate results with 0.8

**Expected impact:**
- H1: 113 correlations @ 0.7 → ~70-80 @ 0.8 (reduction expected)
- More accurate identification of "severe" multicollinearity
- Consistency with literature standards

### ⏳ 5. Phase 02b Univariate Tests - Config Integration

**File:** `scripts/02_exploratory_analysis/02b_univariate_tests.py`

**Required changes:**
1. Import `get_config()`
2. Replace `alpha=0.05` with `config.get('analysis', 'alpha')`
3. Replace `method='fdr_bh'` with `config.get('analysis', 'fdr_method')`
4. Update docstring line 23: "per horizon (64 features each)" NOT "across 64 × 5"

**Impact:** Config-driven, clarified FDR scope.

### ⏳ 6. Phase 01 Scripts - Config Integration

**Files:**
1. `scripts/01_data_preparation/01a_remove_duplicates.py`
   - Use config paths
   - Remove hardcoded "43,405 - 401" (compute dynamically)

2. `scripts/01_data_preparation/01b_outlier_treatment.py`
   - Replace `lower=0.01, upper=0.99` with config values
   - Use config paths

3. `scripts/01_data_preparation/01c_missing_value_imputation.py`
   - Use config for imputation parameters
   - Use config paths

---

## TESTING PLAN

### Step 1: Verify Config Loads

```bash
cd ~/FH-Wedel/WS25/seminar-bankruptcy-prediction
make install
python -c "from src.bankruptcy_prediction.utils.config_loader import get_config; \
           c = get_config(); \
           print(f'VIF threshold: {c.vif_threshold}'); \
           print(f'Corr threshold: {c.get(\"analysis\", \"correlation_threshold\")}'); \
           print(f'Alpha: {c.get(\"analysis\", \"alpha\")}')"
```

**Expected output:**
```
VIF threshold: 10
Corr threshold: 0.8
Alpha: 0.05
```

### Step 2: Re-run Phase 02c (CRITICAL)

```bash
.venv/bin/python scripts/02_exploratory_analysis/02c_correlation_economic.py
```

**Verify:**
- New high correlation counts (should be < 113 for H1)
- HTML says "High Correlations (|r| > 0.8)"  
- No errors

### Step 3: Re-run Phase 03

```bash
.venv/bin/python scripts/03_multicollinearity/03a_vif_analysis.py
```

**Verify:**
- Logs show "Configuration: VIF threshold = 10"
- No NaN/Inf errors (would fail-fast)
- HTML shows dynamic threshold
- Results identical to previous run (threshold was already 10)

### Step 4: Compare Results

```bash
python scripts/paper_helper/verify_phase03_facts.py
```

**Verify:**
- All VIF ≤ 10 still
- Feature counts unchanged (H1=40, H2=41, H3=42, H4=43, H5=41)

---

## DOCUMENTATION TO UPDATE

### 1. PHASE_03_VIF_COMPLETE.md

**Changes needed:**
- Remove claim "per-horizon FDR is more conservative" (it's less conservative)
- Clarify: "Per-horizon FDR appropriate for independent models"
- Add: "BH-FDR with m=64 per horizon vs m=320 pooled"

### 2. docs/PROJECT_STATUS.md

**Changes needed:**
- Update Phase 02c section: "Correlation threshold: 0.8 (standard practice)"
- Update Phase 03 section: "Config-driven thresholds"

### 3. PERFECT_PROJECT_ROADMAP.md

**Changes needed:**
- Note config integration in Phase 02-03 completion status

---

## ACCEPTANCE CRITERIA

All must pass:
- [ ] Config loads without errors
- [ ] All `0.7` replaced with config in 02c
- [ ] All `0.05` / `10` / `0.01`/`0.99` replaced with config
- [ ] Phase 02c re-run completes successfully
- [ ] Phase 02c HTML shows "|r| > 0.8"
- [ ] Phase 02c high correlation counts < previous (more selective)
- [ ] Phase 03 re-run completes successfully
- [ ] Phase 03 HTML shows dynamic threshold
- [ ] No NaN/Inf errors (fail-fast works)
- [ ] FDR documentation corrected
- [ ] All hardcoded paths replaced with config paths
- [ ] Verification script confirms results valid

---

## ROLLBACK PLAN

If anything breaks:
1. Git status to see changes
2. Restore from checkpoint if needed
3. Individual script rollback possible (changes are isolated)

---

**Next action:** Complete Phase 02c refactoring and re-run with 0.8 threshold.

# Prompt to Start Phase 03 (VIF Feature Pruning)

## Objective
Implement Phase 03: multicollinearity control via iterative VIF pruning (threshold 10) per horizon (H1–H5), producing reproducible reports and a final feature set per horizon for downstream modeling.

## Critical Context

**Project Status:**
- **Phase 00 (Foundation):** ✅ Complete
- **Phase 01 (Data Preparation):** ✅ Complete  
- **Phase 02 (Exploratory Analysis):** ✅ **100% COMPLETE (A+ STANDARD)** with FDR correction, D'Agostino K² tests, economic validation
- **Phase 03 (Multicollinearity Control):** ⏳ **TO BE IMPLEMENTED NOW**

**Data Location (verify before starting):**
- Input: `data/processed/poland_imputed.parquet` 
  - Columns: `y` (canonical target), `horizon` (1–5), features `A1..A64`
  - Size: 43,004 observations × 68 columns
  - 0% missing values
  
**Utilities (MUST USE):**
- `src/bankruptcy_prediction/utils/logging_setup.py` — `setup_logging`, `print_header`, `print_section`
- `src/bankruptcy_prediction/utils/target_utils.py` — `get_canonical_target`
- `src/bankruptcy_prediction/utils/metadata_loader.py` — `get_metadata` (for optional tie-breaks)

**Phase 02 Outputs to Consult (read-only):**
- `results/02_exploratory_analysis/02b_H{h}_univariate_tests.xlsx` — effect sizes for tie-breaks
- `results/02_exploratory_analysis/02c_H{h}_correlation.xlsx` — economic plausibility, correlation matrices

**Non-Negotiable Rules (from user memories):**
- ❌ **NO INLINE TERMINAL SCRIPTS** — Always create `.py` files, never `python -c "..."` or here-docs
- ❌ **NO HALLUCINATIONS** — Every number must be verified from actual outputs
- ✅ **ROBUST METHODOLOGY** — Cite primary sources (e.g., O'Brien 2007 for VIF threshold), not Wikipedia
- ✅ **100% REPRODUCIBLE** — Seed everything stochastic (though VIF should be deterministic)
- ✅ **CONSISTENT LOGGING** — Use project's `setup_logging()` utility, create file logs in `logs/`
- ✅ **DEFENSIVE PROGRAMMING** — Handle edge cases (constant columns, perfect multicollinearity)

## Deliverables

### 1. Main Analysis Script
**File:** `scripts/03_multicollinearity/03a_vif_analysis.py`

**Functionality:**
- Read `data/processed/poland_imputed.parquet`, enforce canonical `y` via `get_canonical_target()`
- For **each horizon H1–H5** (process sequentially):
  - Filter data: `df_h = df[df['horizon'] == h]`
  - Extract features: `A1..A64` (64 features initially)
  - **Pre-processing checks:**
    - Drop columns with zero or near-zero variance (`std < 1e-12`)
    - Drop columns with NaN or Inf (should be 0 after Phase 01, but defensive check)
    - Add constant column for VIF computation (`sm.add_constant`)
  - **Iterative VIF pruning:**
    - Compute VIF for all features (exclude constant from removal)
    - If `max(VIF) > 10`: remove feature with highest VIF, record iteration
    - Repeat until `max(VIF) <= 10` OR only 2 features remain (graceful stop)
    - Log each iteration: iteration #, features remaining, max VIF, removed feature
  
**Outputs per horizon:**
- **Excel:** `results/03_multicollinearity/03a_H{h}_vif.xlsx`
  - Sheet `Iterations`: columns [`Iteration`, `Features_Remaining`, `Max_VIF`, `Removed_Feature`, `Removed_VIF`, `Reason`]
  - Sheet `Final_VIF`: columns [`Feature`, `VIF`] sorted by VIF descending
  - Sheet `Removed_Features`: columns [`Feature`, `VIF_at_Removal`, `Iteration_Removed`, `Reason`]
  - Sheet `Metadata`: [`Horizon`, `Initial_Features`, `Final_Features`, `Removed_Count`, `Iterations`, `Max_Final_VIF`, `Timestamp`]
- **HTML:** `results/03_multicollinearity/03a_H{h}_vif.html` — narrative summary + embedded tables (match Phase 02 style)
- **JSON:** `data/processed/feature_sets/H{h}_features.json` — list of final kept features

**Consolidated outputs (all horizons):**
- **Excel:** `results/03_multicollinearity/03a_ALL_vif.xlsx`
  - Sheet `Summary`: one row per horizon with [`Horizon`, `Initial`, `Final`, `Removed`, `Iterations`, `Max_Final_VIF`]
  - Sheet `All_Removed`: all removed features across horizons
- **HTML:** `results/03_multicollinearity/03a_ALL_vif.html` — executive summary
- **Log:** `logs/03a_vif_analysis.log` (via `setup_logging()`)

### 2. Fact Extraction & Verification Script
**File:** `scripts/paper_helper/verify_phase03_facts.py`

**Functionality:**
- Read all `03a_H{h}_vif.xlsx` files (5 horizons)
- Extract verified facts:
  - Per horizon: iterations, removed count, final count, top-5 worst initial VIFs, max final VIF
  - Overall: total removed, average iterations, feature overlap between horizons
- **Output:** `scripts/paper_helper/phase03_facts.json`
- **Validation:** Assert `Initial - Removed = Final` for each horizon
- Print summary to console (for quick verification)
### 3. Optional: Tie-Breaking Logic
**Only implement if identical VIFs occur** (rare but possible with floating-point precision).

**Priority order for removal when VIFs are equal:**
1. **Higher mean correlation** to other features (read from `02c_H{h}_correlation.xlsx`)
2. **Lower economic plausibility** (read from `02c_H{h}_correlation.xlsx` → `Economic_Validation` sheet)
3. **Weaker effect size** (read from `02b_H{h}_univariate_tests.xlsx` → `All_Results` sheet)
4. **Lexicographic** (e.g., A10 before A20) as final fallback

**Documentation:** Record tie-break reason in `Removed_Features` sheet: `"max VIF (tie-break: correlation)"`

### 4. Documentation Updates
After successful implementation:
- Update `PERFECT_PROJECT_ROADMAP.md` — check Phase 03 as complete
- Update `docs/PROJECT_STATUS.md` — add Phase 03 section with file counts and summary
- Create `PHASE_03_VIF_COMPLETE.md` (similar to Phase 02 doc) with:
  - What changed (64 → final count per horizon)
  - Methodology (VIF threshold, iterative pruning)
  - Results summary (per-horizon stats)
  - Code quality notes (logging, utilities, reproducibility)

## Acceptance Criteria

**Per-Horizon Outputs (H1–H5):**
- ✅ Iteration converges to all VIF ≤ 10, OR gracefully stops if ≤ 2 features remain
- ✅ Excel file exists with 4 sheets: `Iterations`, `Final_VIF`, `Removed_Features`, `Metadata`
- ✅ HTML file exists with narrative + embedded tables (consistent with Phase 02 style)
- ✅ JSON file exists with list of final features (e.g., `["A1", "A5", "A12", ...]`)
- ✅ Count consistency: `Initial (64) = Final + Removed`
- ✅ Max VIF in final set ≤ 10.0 (or documented if stopped early)

**Consolidated Outputs:**
- ✅ `03a_ALL_vif.xlsx` with `Summary` sheet (5 rows, one per horizon)
- ✅ `03a_ALL_vif.html` with executive summary
- ✅ Consistent counts across all files

**Code Quality:**
- ✅ Uses `setup_logging()` — creates `logs/03a_vif_analysis.log`
- ✅ Uses `get_canonical_target()` — no hardcoded `'y'` or `'bankrupt'`
- ✅ All `.py` files — **zero inline terminal scripts**
- ✅ Defensive: handles edge cases (constant columns, perfect collinearity, empty sets)
- ✅ Reproducible: deterministic output (VIF is deterministic given data)

**Verification:**
- ✅ `verify_phase03_facts.py` runs without errors
- ✅ `phase03_facts.json` created with validated counts
- ✅ Console prints summary matching Excel metadata

**Documentation:**
- ✅ `PERFECT_PROJECT_ROADMAP.md` updated (Phase 03 checked)
- ✅ `PROJECT_STATUS.md` updated (Phase 03 section added)
- ✅ `PHASE_03_VIF_COMPLETE.md` created (methodology + results)

## Implementation Notes

**VIF Computation (statsmodels):**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add constant (for intercept) BEFORE computing VIF
X_with_const = add_constant(X)

# Compute VIF for each feature (skip constant column)
vif_data = []
for i in range(1, X_with_const.shape[1]):  # Start from 1 to skip constant
    vif = variance_inflation_factor(X_with_const.values, i)
    vif_data.append({'Feature': X.columns[i-1], 'VIF': vif})
```

**Edge Case Handling:**
- **Perfect multicollinearity:** If VIF = inf, remove immediately (before iterating)
- **Near-zero variance:** Drop features with `std < 1e-12` before VIF computation
- **NaN/Inf in data:** Should not exist after Phase 01, but check defensively
- **Convergence failure:** If stuck after 100 iterations, log warning and stop gracefully

**VIF Threshold Justification:**
- Standard threshold: **VIF > 10** indicates severe multicollinearity (O'Brien 2007; Marquaridt 1970)
- Alternative thresholds (5, 4) are more conservative but would remove too many features
- Document in code comments and HTML output

**HTML Styling (match Phase 02):**
- Use same CSS classes and table formatting as `02a/b/c` HTML outputs
- Include metadata footer with timestamp and script name
- Professional, clean layout suitable for paper appendix

## Run Commands

**Recommended execution sequence:**
```bash
# 1. Activate environment and sync dependencies
make install

# 2. Verify data exists (defensive check)
ls -lh data/processed/poland_imputed.parquet

# 3. Run VIF analysis (creates all outputs)
./.venv/bin/python scripts/03_multicollinearity/03a_vif_analysis.py

# 4. Verify outputs exist
ls -lh results/03_multicollinearity/
ls -lh data/processed/feature_sets/
cat logs/03a_vif_analysis.log | tail -50

# 5. Extract and validate facts
./.venv/bin/python scripts/paper_helper/verify_phase03_facts.py

# 6. Review facts
cat scripts/paper_helper/phase03_facts.json
```

## Expected Output Structure

After successful execution, directory structure should be:
```
results/03_multicollinearity/
├── 03a_H1_vif.xlsx
├── 03a_H1_vif.html
├── 03a_H2_vif.xlsx
├── 03a_H2_vif.html
├── 03a_H3_vif.xlsx
├── 03a_H3_vif.html
├── 03a_H4_vif.xlsx
├── 03a_H4_vif.html
├── 03a_H5_vif.xlsx
├── 03a_H5_vif.html
├── 03a_ALL_vif.xlsx
└── 03a_ALL_vif.html

data/processed/feature_sets/
├── H1_features.json
├── H2_features.json
├── H3_features.json
├── H4_features.json
└── H5_features.json

scripts/paper_helper/
└── phase03_facts.json

logs/
└── 03a_vif_analysis.log
```

**Total files created:** 18 (10 Excel/HTML + 5 JSON feature sets + 1 facts JSON + 1 log + 1 completion doc)

## Success Checklist

After implementation, verify:
- [ ] All 18 files created
- [ ] Log file shows 5 horizons processed successfully
- [ ] Each horizon has: Excel (4 sheets), HTML (narrative), JSON (feature list)
- [ ] Consolidated Excel has summary of all 5 horizons
- [ ] `phase03_facts.json` shows:
  - Per horizon: `initial: 64`, `final: X`, `removed: Y`, `iterations: Z`, `max_final_vif: ≤10.0`
  - Overall: `total_removed: sum(Y)`, `avg_iterations: mean(Z)`
- [ ] Feature counts make sense (not all removed, not zero removed)
- [ ] No errors in log file
- [ ] HTML outputs render correctly in browser
- [ ] Ready to write Chapter 6 section based on `phase03_facts.json`

## Clarifications

**If anything is unclear, ASK FIRST before implementation.**

Otherwise: 
1. Implement `03a_vif_analysis.py` exactly as specified
2. Implement `verify_phase03_facts.py` 
3. Run both scripts
4. Create `PHASE_03_VIF_COMPLETE.md` documentation
5. Update roadmap and project status
6. Report back with:
   - ✅ Success confirmation
   - 📊 Quick stats from `phase03_facts.json` (final feature counts per horizon)
   - 🔍 Any unexpected findings (e.g., very few or very many features removed)
   - 📁 File count verification (should be 18 new files)

**Remember:** NO HALLUCINATIONS. Every number in reports must come from actual computation, not assumptions!
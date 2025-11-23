# Project Status (Updated) – Bankruptcy Prediction Seminar

**Date:** 2025-11-20  
**Scope of Update:** Incorporates progress through Feature Selection (Phase 04) and Modeling integration (Phase 05 partial), including CatBoost CV integration, discrepancy explanation, and PR-AUC reporting.

---

## Executive Snapshot

| Phase | Status | Scripts (Executed/Planned) | Key Outputs | Notes |
|-------|--------|----------------------------|-------------|-------|
| 00 Foundation | ✅ Complete | 4/4 | results/00_foundation/* | Data understanding & quality issues documented |
| 01 Data Preparation | ✅ Complete | 3/3 | poland_imputed.parquet | Duplicates removed, winsorization, MICE imputation |
| 02 Exploratory Analysis | ✅ Complete | 3/3 | results/02_exploratory/* | Distribution, univariate tests, correlation matrices |
| 03 Multicollinearity | ✅ Complete | 1/1 | results/03_multicollinearity/* | VIF pruning → 40–43 features/horizon |
| 04 Feature Selection | ✅ Complete | 4/4 | results/04_feature_selection/* | Filter + Wrapper + Embedded + Consensus |
| 05 Modeling | ▶ In Progress | 3/5 (baseline, extra, CatBoost) | results/05_modeling/* | Ensemble + CatBoost CV added; PR-AUC surfaced |
| 06 Evaluation | ⏳ Pending | 0/1 | — | To add calibration & threshold diagnostics |
| 07 Paper Writing | 📝 Ongoing | — | seminar-paper/*.tex | Chapter 08 updated (discrepancy explanation + PR-AUC table) |

**Overall Completion (scripts):** 18 / ~22 (≈82%)  
**Primary Performance Metrics (current best per horizon, 5-fold Stratified CV):**

| Horizon | Best Model | ROC-AUC | PR-AUC | Features Used |
|---------|------------|---------|--------|---------------|
| H1 | softvote_lr_gb_et | 0.796 | 0.183 | 10 |
| H2 | random_forest | 0.845 | 0.341 | 9 |
| H3 | random_forest | 0.780 | 0.140 | 7 |
| H4 | random_forest | 0.812 | 0.218 | 8 |
| H5 | random_forest | 0.864 | 0.420 | 10 |

Source: `seminar-paper/tables/phase05_modeling_summary.tex` (auto-generated).

---

## Key Advances Since Previous Status

1. **Phase 04 Full Completion:** Wrapper (04b), Embedded (04c), Consensus (04d) executed; consensus reports present (`04d_ALL_consensus.{html,xlsx}`).  
2. **Modeling Enhancements:** Added CatBoost under the same 5-fold CV regime (script `05c_modeling_catboost.py`) alongside baseline and extra models.  
3. **Ensemble Integration:** Soft voting (LR + GB + ET) improves H1 marginally (best ROC-AUC 0.796).  
4. **Metric Expansion:** PR-AUC now included in summary LaTeX table (update to `generate_phase05_tables.py`).  
5. **Documentation Alignment:** Chapter 08 (`08_Modellierung.tex`) extended with discrepancy explanation (historic single-split CatBoost ≈0.9812 vs current cross-validated results).  
6. **Consistency With Methodology:** All modeling now uses stratified 5-fold CV (uniform across horizons) mitigating optimistic bias of archived single temporal split.  
7. **Class Imbalance Handling:** All models utilize `class_weight='balanced'` (and CatBoost weight mapping), improving PR-AUC interpretability.  
8. **LaTeX Build Stability:** Table row terminators fixed; full paper compiles without fatal errors (only cosmetic overfull boxes).

---

## Discrepancy Clarification (Historical vs Current Performance)

| Aspect | Archived Implementation | Current Implementation |
|--------|------------------------|------------------------|
| Data Split | Single temporal train/test | 5-fold stratified CV |
| Feature Set | All / broader | EPV-conform reduced sets |
| Metric Surfacing | ROC-AUC only emphasized | ROC-AUC + PR-AUC reported |
| Potential Leakage | Higher risk (no multistage control) | Controlled (VIF + selection + CV) |
| Reported Peak AUC | CatBoost ≈0.9812 | 0.864 (H5 RF) |

Result: Lower but more conservative and generalizable metrics now documented.

---

## Remaining Work (Targeted)

| Area | Task | Priority | Notes |
|------|------|----------|-------|
| Phase 05 | Add threshold / calibration analysis | High | Brier score, reliability plots |
| Phase 05 | Feature importance exports (RF, CatBoost) | Medium | SHAP optional (performance) |
| Phase 06 | Unified evaluation dashboard | High | Aggregate CV vs temporal holdout (optional) |
| Paper | Expand Chapter 08 with calibration & threshold rationale | Medium | After evaluation scripts |
| Paper | Add brief PR-AUC interpretation section | Medium | Imbalance justification |
| Reproducibility | Parameter YAML for CatBoost | Low | Mirror RF documented params |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Class imbalance lowers PR-AUC stability | Medium | Maintain class weights, consider threshold tuning |
| Overfitting with small feature sets | Medium | Nested CV for feature selection already in place; consider external validation if time permits |
| Documentation drift (legacy vs updated) | High | This file supersedes `PROJECT_STATUS.md`; retain original as historical snapshot |
| Time for Phase 06 evaluation | Medium | Scope threshold & calibration to core models only |

---

## File Inventory (New / Modified in This Iteration)

- `scripts/05_modeling/05c_modeling_catboost.py` – CatBoost CV integration
- `scripts/paper_helper/generate_phase05_tables.py` – Added PR-AUC column & LaTeX fixes
- `seminar-paper/tables/phase05_modeling_summary.tex` – Updated table including PR-AUC
- `seminar-paper/kapitel/08_Modellierung.tex` – Added discrepancy explanation subsection

---

## Validation Checklist

| Item | Status |
|------|--------|
| CatBoost metrics written to `H*_metrics.json` | ✅ |
| PR-AUC present in modeling summary table | ✅ |
| Paper compiles (`make paper`) | ✅ |
| Consensus feature selection outputs exist | ✅ (`04d_ALL_consensus.*`) |
| Discrepancy explanation in Chapter 08 | ✅ |
| Updated status file created | ✅ (this file) |

---

## Supersession Note

`docs/PROJECT_STATUS.md` is retained as a historical snapshot (pre-Phase 04/05 modeling). **Authoritative current status = this file.**

---

## Next Immediate Command Suggestions

```bash
# Optional: generate evaluation/plots once Phase 06 scripts exist
make phase05-modeling        # Re-run baseline (if features change)
make phase05-modeling-extra  # Re-run extra models / ensemble
python scripts/05_modeling/05c_modeling_catboost.py  # Re-run CatBoost CV
make phase05-tables          # Refresh LaTeX tables
make paper                   # Recompile with latest results
```

---

**Maintained by:** Automated assistant (GitHub Copilot) – Session documentation aligned to 2025-11-20 changes.


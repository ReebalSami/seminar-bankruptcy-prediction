# Bankruptcy Prediction - Seminar Project

**Institution:** FH Wedel  
**Semester:** WS 2024/25  
**Topic:** Entwicklung eines Frühwarnsystems für Unternehmenskrisen mit Hilfe maschinellen Lernens  
**Goal:** German grade 1.0 (excellent)

---

## Project Overview

**Research Focus:**
- Early warning indicators for corporate bankruptcy
- Predictive analytics using financial ratios
- Comparison: Random Forests vs Logistic Regression
- Multi-horizon prediction (1-5 years ahead)

**Dataset:** Polish Companies Bankruptcy (Kaggle)
- 43,405 observations
- 64 financial ratio features (A1-A64)
- 5 prediction horizons (H1-H5)
- Target: Bankruptcy within horizon period

---

## Status & Results (Consolidated)

- **Phases 00–03 (Foundation, Prep, EDA, Multikollinearität):** abgeschlossen. Datensatz: 43.405 Beobachtungen, 64 Kennzahlen, starke Klassenungleichverteilung (4,82%). Entscheidung: horizontspezifische Modelle (H1–H5).
- **Phase 04 (Feature Selection):**
  - Alte (nicht-verschachtelte) Auswahl für H1–H5 als Grundlage der Modellierung verwendet; Guardrails: EPV ≥ 10, Leistungs-Retention.
  - Verschachtelte (Nested-CV) Ergebnisse vorhanden für H1–H3; H4/H5 in Arbeit. Beobachtung: AUC-Differenzen klein; Nested reduziert Optimismus, v. a. bei Ridge.
  - Stabilität (Nogueira): H2 Ridge > Lasso > EN; H3 Lasso > Ridge > EN. Konsensbildung mit EPV-Deckelung.
- **Phase 05/06 (Modellierung & Evaluation; 5-fach Stratified CV, class_weight=balanced):**
  - ROC-AUC Sieger je Horizont: H1 Soft-Voting 0,796; H2–H5 Random Forest 0,845 / 0,780 / 0,812 / 0,864.
  - PR-AUC der Sieger: H1 0,183; H2 0,341; H3 0,140; H4 0,218; H5 0,420.
- **Seminar-Paper:** LaTeX-Kapitel 07–09 aktualisiert (Nested vs. Alt, Stabilität, Future Work). Tabellen (Phase 04/05) werden aus Ergebnissen generiert.

### Makefile Quickstart
- Setup: `make install`
- Phase 04 Tabellen + Paper: `make phase04-tables && make phase05-tables && make paper`
- Modeling (falls erneut nötig): `make phase05-modeling` und `make phase05-modeling-extra`, danach `make phase05-eval` und `make phase05-tables`.

### Paper Build
- PDF: `seminar-paper/doku_main.pdf`
- Tabellenquellen: `seminar-paper/tables/phase04_*` und `phase05_*`

### Methodische Guardrails
- EPV-Regel (min. 10 Events je Variable) und Leistungs-Retention in 04d umgesetzt.
- Nested-CV für eingebettete Methoden (H1–H3), Stabilität nach Nogueira berichtet.

### Bekannte Limitationen
- Mögliche Panel-/Identitätsleckage ohne Firmen-ID; Empfehlung: StratifiedGroupKFold sobald Gruppen verfügbar.
- Keine Kalibrierung/Schwellenoptimierung in Phase 05/06 (bewusst ausgelassen; siehe Future Work).
- H4–H5: Nested-CV noch laufend; Modellierung erfolgt fürs Paper mit alter Auswahl, Nested als Sensitivitätsvergleich.

### Future Work (Kurzüberblick)
- Group-aware CV (StratifiedGroupKFold), horizontspezifisches Tuning, Kalibrierung (Platt/Isoton) und Schwellenanalyse, Zeit-/Block-CV, Stabilitäts-gewichtete Konsensbildung, Cross-Dataset-Vergleich.
**Planned Activities:**
- Distribution analysis per horizon (H1-H5)
- Univariate feature analysis (t-tests, effect sizes)
- Correlation matrices and multicollinearity checks
- Feature importance rankings

**Research-Backed Sequence:**
```
Remove duplicates → Treat outliers → Impute missing values → Scale → Split
```

**Why this order:** "When outliers are removed or treated and missing values accurately imputed, the correlations among predictors become more realistic" (Number Analytics, 2024)

---

## Project Structure

```
seminar-bankruptcy-prediction/
├── scripts/
│   ├── 00_foundation/       # Dataset understanding (COMPLETE)
│   ├── 01_data_preparation/ # Cleaning & preprocessing (NEXT)
│   ├── 02_exploratory/      # EDA on cleaned data
│   ├── 03_multicollinearity/# VIF analysis & feature reduction
│   ├── 04_feature_selection/# Feature importance & selection
│   └── 05_model_evaluation/ # Modeling & evaluation
│
├── results/
│   └── 00_foundation/       # Excel, HTML, PNG outputs
│
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned data
│
└── src/bankruptcy_prediction/  # Shared utilities
    ├── data/                   # Data loaders
    ├── features/               # Feature engineering
    └── utils/                  # Logging, config
```

---

## Key Methodological Decisions

### ✅ **What We Got Right**

1. **Analysis-First Approach:** Understand data before preprocessing
2. **Foundation Phase:** Complete characterization of dataset
3. **Evidence-Based:** Research citations for all decisions
4. **Honest Reporting:** Document limitations and assumptions

### ⚠️ **Critical Issues Identified**

1. **Duplicate Nature Unknown:**
   - 401 exact duplicates (all 68 columns identical)
   - NO company ID → can't determine if same company or error
   - **Assumption:** Data entry errors → remove in Phase 01

2. **Horizon Heterogeneity:**
   - Bankruptcy rate: 3.86% (H1) → 6.94% (H5) = **80% increase**
   - Foundation analyzed ALL horizons combined
   - **Decision needed:** Separate models OR pooled with horizon feature

3. **Incomplete Initial Analysis:**
   - Script 00d v1: Only 10/64 features for outliers ❌
   - Script 00d v2: ALL 64 features analyzed ✅

---

## Running the Project

### Setup
```bash
make install  # Activate venv and sync dependencies
```

### Execute Foundation Scripts
```bash
python scripts/00_foundation/00a_polish_dataset_overview.py
python scripts/00_foundation/00b_polish_feature_analysis.py
python scripts/00_foundation/00c_polish_temporal_structure.py
python scripts/00_foundation/00d_polish_data_quality.py
```

### View Results
```bash
open results/00_foundation/00a_polish_overview.html
open results/00_foundation/00d_data_quality.xlsx
```

---

## Archived documentation

All standalone Markdown documentation has been consolidated into this README. Historical files were moved to `archive/docs/` (and existing `archive/*` subfolders). Refer to the LaTeX paper for canonical methodology and results.

---

## Research Methodology

**Preprocessing Pipeline Order (Evidence-Based):**
1. Remove duplicates (prevent leakage)
2. Treat outliers (3×IQR winsorization)
3. Impute missing values (passive imputation for ratios)
4. Scale features (z-score normalization)
5. Split data (temporal holdout: H1-H3 / H4 / H5)

**Then:**
6. Calculate VIF (requires complete data)
7. Remove multicollinear features (VIF > 10)
8. Feature selection (forward/backward, importance)
9. Model training (Logistic, Random Forest, XGBoost)
10. Evaluation (ROC-AUC, PR-AUC, calibration)

**Critical:** VIF analysis MUST come AFTER imputation (research-backed)

---

## Professor's Criteria (Grade 1.0)

✅ **Methodology:** Evidence-based, research citations  
✅ **Honesty:** Document failures and assumptions  
✅ **Completeness:** 30-40 pages, all phases covered  
✅ **Econometrics:** Proper GLM diagnostics (not OLS)  
✅ **Validation:** Temporal holdout, no data leakage  

---

## References

1. Number Analytics (2024). "VIF Strategies: Reducing Multicollinearity"
2. Von Hippel (2013). "Multiple imputation for ratios." *Statistics in Medicine*
3. Coats & Fant (1993). "Bankruptcy prediction across time horizons"
4. Feature-engine documentation. "Missing Data Imputation"

---

**Status:** Foundation phase validated. Ready for Phase 01 after strategy decision.

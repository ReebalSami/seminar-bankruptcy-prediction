# 🔧 REPOSITORY REORGANIZATION PLAN

**Date:** November 13, 2025  
**Problem:** Messy repo, unclear script sequence, uncertainty about what loads what  
**Solution:** Complete reorganization with clear execution order and verification

---

## 🚨 CURRENT PROBLEMS

1. ❌ Scripts added ad-hoc without clear numbering (02b, 02c, etc.)
2. ❌ Unclear dependencies (which script needs which output?)
3. ❌ VIF analysis came AFTER modeling (should be BEFORE!)
4. ❌ Multiple verification scripts (VERIFY, VISUALIZE, ANALYZE, CHECK)
5. ❌ Unclear what's foundation vs what's analysis
6. ❌ Can't easily see execution order

---

## ✅ REORGANIZATION PRINCIPLES

### **1. Foundation First (Script 00)**
- Cross-dataset feature mapping
- Temporal structure verification
- **MUST run before anything else**

### **2. Dataset-Specific Pipelines (Scripts 01-08)**
- Each dataset gets own directory
- Clear numeric sequence
- VIF BEFORE modeling

### **3. Cross-Dataset Analysis (Scripts 09+)**
- Transfer learning
- Comparative analysis
- **ONLY after individual datasets done**

### **4. One Script = One Purpose**
- No duplicate verification scripts
- Clear naming
- Documented inputs/outputs

---

## 📁 PROPOSED NEW STRUCTURE

```
scripts/
├── 00_foundation/                     ← RUN FIRST!
│   ├── 00_cross_dataset_feature_mapping.py
│   ├── 00_verify_mapping.py
│   └── README.md                      ← What runs when
│
├── 01_polish/                         ← Polish pipeline
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_vif_remediation.py          ← BEFORE modeling!
│   ├── 04_baseline_models.py          ← Uses VIF output
│   ├── 05_advanced_models.py
│   ├── 06_model_calibration.py
│   ├── 07_robustness.py
│   ├── 08_summary.py
│   ├── RUN_ALL_POLISH.sh
│   └── README.md
│
├── 02_american/                       ← American pipeline
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_vif_remediation.py          ← BEFORE modeling!
│   ├── 04_regularization_comparison.py ← Handle high VIF
│   ├── 05_baseline_models.py          ← Uses regularization
│   ├── 06_advanced_models.py
│   ├── 07_calibration.py
│   ├── 08_summary.py
│   ├── RUN_ALL_AMERICAN.sh
│   └── README.md
│
├── 03_taiwan/                         ← Taiwan pipeline
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_vif_remediation.py          ← BEFORE modeling!
│   ├── 04_baseline_models.py          ← Uses VIF output
│   ├── 05_advanced_models.py
│   ├── 06_calibration.py
│   ├── 07_robustness.py
│   ├── 08_summary.py
│   ├── RUN_ALL_TAIWAN.sh
│   └── README.md
│
└── 04_cross_dataset/                  ← RUN LAST!
    ├── 01_transfer_learning.py        ← Uses Script 00 mappings
    ├── 02_comparative_analysis.py
    ├── 03_ensemble.py
    ├── RUN_ALL_TRANSFER.sh
    └── README.md
```

---

## 🔄 MIGRATION PLAN - STEP BY STEP

### **Phase 1: Create Clean Structure (30 min)**

1. Create new directories
2. Move/rename scripts to correct locations
3. Delete duplicates and verification scripts
4. Create README in each directory

### **Phase 2: Fix Script Dependencies (1 hour)**

For each script, document:
- **Inputs:** What files does it read?
- **Outputs:** What files does it create?
- **Dependencies:** What must run before it?

### **Phase 3: Test Execution Order (2 hours)**

Run each pipeline sequentially:
1. Script 00 (foundation)
2. Polish 01-08
3. American 01-08
4. Taiwan 01-08
5. Cross-dataset 01-03

Verify at each step that outputs are correct.

### **Phase 4: Create Master Runner (30 min)**

Create `RUN_ENTIRE_PROJECT.sh` that runs everything in order.

---

## 📋 DETAILED MIGRATION STEPS

### **Step 1: Foundation Scripts (Script 00)**

**Current:**
```
scripts/00_foundation/00_cross_dataset_feature_mapping.py    ← OLD
scripts/00_foundation/00_FIXED_cross_dataset_feature_mapping.py  ← NEW
scripts/VERIFY_FEATURE_MAPPING.py                            ← Standalone
scripts/VISUALIZE_MAPPING.py                                 ← Standalone
scripts/CHECK_SEMANTIC_VIF.py                                ← Standalone
```

**Action:**
```bash
# Keep only the FIXED version
mv scripts/00_foundation/00_FIXED_cross_dataset_feature_mapping.py \
   scripts/00_foundation/00_cross_dataset_feature_mapping.py

# Delete old version
rm scripts/00_foundation/00_cross_dataset_feature_mapping.py (old)

# Move verification into foundation
mv scripts/VERIFY_FEATURE_MAPPING.py \
   scripts/00_foundation/00_verify_mapping.py

# Delete redundant scripts
rm scripts/VISUALIZE_MAPPING.py
rm scripts/CHECK_SEMANTIC_VIF.py
rm scripts/ANALYZE_FEATURE_USAGE.py
```

**Result:**
```
scripts/00_foundation/
├── 00_cross_dataset_feature_mapping.py    ← Main script
├── 00_verify_mapping.py                   ← Verification
└── README.md                               ← Explains what/when
```

---

### **Step 2: Polish Scripts (01_polish)**

**Current:**
```
01_data_understanding.py
02_exploratory_analysis.py
03_data_preparation.py
04_baseline_models.py           ← Uses what features?
05_advanced_models.py
06_model_calibration.py
07_robustness_analysis.py
08_econometric_analysis.py
10c_glm_diagnostics.py          ← Out of order!
10d_remediation_save_datasets.py ← Out of order!
11_temporal_holdout_validation.py
12_cross_dataset_transfer.py
12_cross_dataset_transfer_SEMANTIC.py
13c_temporal_validation.py
```

**Action:**
```bash
# Rename to correct order
mv 10d_remediation_save_datasets.py 03_vif_remediation.py

# Keep current 04-08 (but verify they use VIF output!)

# Move cross-dataset scripts to 04_cross_dataset/
mv 12_cross_dataset_transfer.py ../04_cross_dataset/01_transfer_learning.py
mv 12_cross_dataset_transfer_SEMANTIC.py (delete - use 01_transfer_learning.py)

# Delete/archive diagnostics that are now in 03
rm 10c_glm_diagnostics.py (or merge into 03)
rm 11_temporal_holdout_validation.py (or make it 09_temporal_validation.py)
rm 13c_temporal_validation.py (duplicate?)
```

**Result:**
```
scripts/01_polish/
├── 01_data_understanding.py
├── 02_exploratory_analysis.py
├── 03_vif_remediation.py          ← VIF BEFORE modeling
├── 04_baseline_models.py          ← Loads VIF output
├── 05_advanced_models.py
├── 06_model_calibration.py
├── 07_robustness_analysis.py
├── 08_summary.py
└── README.md
```

---

### **Step 3: American Scripts (02_american)**

**Current:**
```
01_data_cleaning.py
02_eda.py
02b_vif_multicollinearity_remediation.py   ← Good!
02c_regularization_with_multicollinearity.py ← Good!
03_baseline_models.py                       ← Does it use VIF?
04_advanced_models.py
05_model_calibration.py
06_feature_importance.py
07_robustness_check.py
08_summary_report.py
```

**Action:**
```bash
# Rename to logical order
mv 02b_vif_multicollinearity_remediation.py 03_vif_remediation.py
mv 02c_regularization_with_multicollinearity.py 04_regularization_comparison.py

# Renumber subsequent scripts
mv 03_baseline_models.py 05_baseline_models.py
mv 04_advanced_models.py 06_advanced_models.py
mv 05_model_calibration.py 07_calibration.py
mv 06_feature_importance.py (merge into 05 or 06)
mv 07_robustness_check.py (merge into 06)
mv 08_summary_report.py 08_summary.py
```

**Result:**
```
scripts/02_american/
├── 01_data_cleaning.py
├── 02_eda.py
├── 03_vif_remediation.py          ← VIF analysis
├── 04_regularization_comparison.py ← Handle high VIF
├── 05_baseline_models.py          ← Uses regularization
├── 06_advanced_models.py
├── 07_calibration.py
├── 08_summary.py
└── README.md
```

---

### **Step 4: Taiwan Scripts (03_taiwan)**

**Current:**
```
01_data_cleaning.py
02_eda.py
02b_vif_multicollinearity_remediation.py
03_baseline_models.py
04_advanced_models.py
05_calibration.py
06_feature_importance.py
07_robustness.py
08_summary.py
```

**Action:**
```bash
# Rename VIF script
mv 02b_vif_multicollinearity_remediation.py 03_vif_remediation.py

# Renumber subsequent scripts
mv 03_baseline_models.py 04_baseline_models.py
# Rest stays same (already correct!)
```

**Result:**
```
scripts/03_taiwan/
├── 01_data_cleaning.py
├── 02_eda.py
├── 03_vif_remediation.py          ← VIF BEFORE modeling
├── 04_baseline_models.py          ← Uses VIF output
├── 05_advanced_models.py
├── 06_calibration.py
├── 07_robustness.py
├── 08_summary.py
└── README.md
```

---

### **Step 5: Create Cross-Dataset Directory**

**Action:**
```bash
# Create new directory
mkdir -p scripts/04_cross_dataset

# Move transfer learning
mv scripts/01_polish/12_cross_dataset_transfer_SEMANTIC.py \
   scripts/04_cross_dataset/01_transfer_learning.py

# Create comparative analysis (new)
# Create ensemble (new)
```

**Result:**
```
scripts/04_cross_dataset/
├── 01_transfer_learning.py        ← Uses Script 00 mappings
├── 02_comparative_analysis.py     ← Compare all 3 datasets
├── 03_ensemble.py                 ← Ensemble all models
└── README.md
```

---

## 📝 README TEMPLATES

### **scripts/README.md (Master)**

```markdown
# Scripts Execution Order

## Phase 1: Foundation (MUST RUN FIRST)
```bash
cd scripts/00_foundation
./RUN_FOUNDATION.sh
```

## Phase 2: Individual Datasets (Can run in parallel)
```bash
cd scripts/01_polish && ./RUN_ALL_POLISH.sh &
cd scripts/02_american && ./RUN_ALL_AMERICAN.sh &
cd scripts/03_taiwan && ./RUN_ALL_TAIWAN.sh &
wait
```

## Phase 3: Cross-Dataset Analysis (After Phase 2)
```bash
cd scripts/04_cross_dataset
./RUN_ALL_TRANSFER.sh
```

## Complete Pipeline
```bash
./RUN_ENTIRE_PROJECT.sh  # Runs all phases in order
```
```

### **scripts/01_polish/README.md (Example)**

```markdown
# Polish Dataset Pipeline

## Execution Order
1. `01_data_understanding.py` → Outputs: EDA report
2. `02_exploratory_analysis.py` → Outputs: Plots
3. `03_vif_remediation.py` → **CRITICAL: Creates VIF-selected datasets**
   - Output: `poland_h1_vif_selected.parquet`
   - Output: `poland_h1_forward_selected.parquet`
4. `04_baseline_models.py` → **Loads:** VIF datasets
5. `05_advanced_models.py`
6. `06_model_calibration.py`
7. `07_robustness_analysis.py`
8. `08_summary.py`

## Quick Run
```bash
./RUN_ALL_POLISH.sh
```

## Dependencies
- Requires: Script 00 (foundation) completed
- Outputs used by: Script 04 (cross-dataset)
```

---

## ✅ VERIFICATION CHECKLIST

After reorganization, verify each step:

### **Foundation (Script 00):**
- [ ] `00_cross_dataset_feature_mapping.py` runs without errors
- [ ] Creates `feature_semantic_mapping_FIXED.json`
- [ ] Creates `common_features.json`
- [ ] Verification script confirms 10/10 features exist

### **Polish Pipeline:**
- [ ] Scripts run in order 01→08
- [ ] 03 creates VIF datasets BEFORE 04 uses them
- [ ] 04-08 load from VIF datasets (not full dataset!)
- [ ] Final summary shows AUC ~0.78-0.83

### **American Pipeline:**
- [ ] Scripts run in order 01→08
- [ ] 03 creates VIF analysis (even if only 2 features!)
- [ ] 04 compares regularization methods
- [ ] 05-08 use regularized models
- [ ] Final summary shows AUC ~0.68

### **Taiwan Pipeline:**
- [ ] Scripts run in order 01→08
- [ ] 03 creates VIF datasets with 5-22 features
- [ ] 04-08 load from VIF datasets
- [ ] Final summary shows AUC ~0.88-0.91

### **Cross-Dataset:**
- [ ] 01 loads Script 00 mappings
- [ ] Transfer learning uses 10 semantic features
- [ ] Results show improvement over positional matching

---

## 🕐 TIMELINE

| Phase | Time | What |
|-------|------|------|
| **Phase 1** | 30 min | Create directories, move files |
| **Phase 2** | 1 hour | Fix dependencies, update imports |
| **Phase 3** | 2 hours | Test each pipeline sequentially |
| **Phase 4** | 30 min | Create master runner & READMEs |
| **Total** | **4 hours** | Complete reorganization |

---

## 🎯 BENEFITS AFTER REORGANIZATION

1. ✅ Clear execution order (00 → 01-03 → 04)
2. ✅ VIF BEFORE modeling (correct methodology!)
3. ✅ No duplicate scripts
4. ✅ Easy to verify what loads what
5. ✅ Can run entire project with one command
6. ✅ Professor can easily follow pipeline
7. ✅ Maintainable and professional

---

## 🚀 NEXT STEPS

**IMMEDIATE (Do Now):**
1. Review this plan
2. Confirm structure makes sense
3. Start Phase 1 (create directories)

**THEN (Sequential):**
1. Phase 1: Move files
2. Phase 2: Fix imports
3. Phase 3: Test pipelines
4. Phase 4: Documentation

**I will help you step-by-step, verifying each phase before moving to next!**

---

**Status:** 📋 Plan ready, awaiting your approval to start  
**Generated:** November 13, 2025, 10:45 AM

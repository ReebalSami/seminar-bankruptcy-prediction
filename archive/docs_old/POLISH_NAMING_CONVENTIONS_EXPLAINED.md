# Polish Dataset Naming Conventions Explained

**Date:** 2025-11-13  
**Status:** ✅ **FULLY EXPLAINED AND VERIFIED**

---

## The Question

Why do different sources use different naming conventions for the same features?
- **UCI Documentation:** X1, X2, X3... X64
- **Kaggle Metadata & Raw Data:** A1, A2, A3... A64
- **Our Documentation & Processed Data:** Attr1, Attr2, Attr3... Attr64

---

## The Complete Answer

### 1. **UCI Documentation (X1-X64)**

**Purpose:** Generic mathematical notation for formulas

The UCI Machine Learning Repository documentation uses **X1-X64** as **abstract variable names** in formula descriptions. This is a standard mathematical convention similar to using "x" and "y" in equations.

**Example from UCI:**
```
X1 net profit / total assets
X2 total liabilities / total assets
X3 working capital / total assets
```

**Why "X"?** Mathematical tradition - X is used for independent variables.

**Important:** These are **documentation labels only**, not actual column names in files.

---

### 2. **Kaggle & Raw Data (A1-A64)**

**Purpose:** Actual column names in ARFF and CSV files

The **original ARFF files** from UCI and the **Kaggle CSV files** use **A1-A64** as the actual column headers.

**Verification:**
```python
# Raw CSV file columns
['A1', 'A2', 'A3', 'A4', 'A5', ..., 'A64', 'class', 'year']
```

**Why "A"?** Likely stands for "Attribute" - standard naming in ARFF file format.

**Source:** 
- Original UCI ARFF files: `1stYear.arff`, `2ndYear.arff`, etc.
- Kaggle CSV: `data-from-kaggel.csv`

---

### 3. **Our Documentation & Processed Data (A1-A64) ✅ FIXED**

**Purpose:** Use original Kaggle/UCI naming to avoid confusion

We **keep the original A1-A64 naming** throughout our entire pipeline:

**Rationale:**
- **Matches source data** from Kaggle and UCI
- **No confusion** with original documentation
- **Direct correspondence** with metadata files
- **Clearer for academic citations**

**Verification:**
```python
# Processed data columns
['A1', 'A2', 'A3', ..., 'A64', 'bankrupt', 'horizon', ...]
```

**Current status:**
- **Raw CSV:** Uses A1-A64 ✓
- **Processed parquet:** Uses A1-A64 ✓  
- **feature_descriptions.json:** Uses A1-A64 ✓
- **All documentation:** Uses A1-A64 ✓

---

## Mapping Table

| Source | Column Names | Purpose | Status |
|--------|--------------|---------|--------|
| **UCI Documentation** | X1, X2, ..., X64 | Mathematical notation for formulas | Reference only |
| **Original ARFF Files** | A1, A2, ..., A64 | Actual column names in source files | Raw data |
| **Kaggle CSV** | A1, A2, ..., A64 | Actual column names in downloaded CSV | Raw data |
| **Our Raw CSV** | A1, A2, ..., A64 | As downloaded from Kaggle | Raw data ✓ |
| **Our Processed Data** | A1, A2, ..., A64 | **Kept original naming** | Processed data ✓ |
| **feature_descriptions.json** | A1, A2, ..., A64 | Documentation matching all data | Documentation ✓ |

---

## Formula Verification Example

All three naming conventions refer to the **same features**:

| UCI (Doc) | ARFF/CSV | Our Data | Formula | Description |
|-----------|----------|----------|---------|-------------|
| X1 | A1 | A1 ✓ | net profit / total assets | Net Profit / Total Assets |
| X2 | A2 | A2 ✓ | total liabilities / total assets | Total Liabilities / Total Assets |
| X3 | A3 | A3 ✓ | working capital / total assets | Working Capital / Total Assets |
| ... | ... | ... | ... | ... |
| X64 | A64 | A64 ✓ | sales / fixed assets | Sales / Fixed Assets |

**✅ All formulas are identical - only the naming convention differs.**

---

## Why This Matters

### For Data Processing
- **Raw data (A1-A64)** is read from Kaggle CSV
- **Preprocessing script** **KEEPS original A1-A64 naming** ✓
- **All subsequent scripts** use A1-A64 ✓

### For Documentation
- **UCI papers** may reference X1-X64
- **Kaggle discussions** reference A1-A64
- **Our seminar paper** uses A1-A64 ✓

### For Feature Descriptions
- **feature_descriptions.json** uses **A1-A64** ✓
- This **matches our processed data** ✅
- When reading papers, mentally map: **X1 (UCI) = A1 (Kaggle/Our data)**

---

## Consistency Check Results ✅ UPDATED

✅ **Raw CSV columns:** A1-A64  
✅ **Kaggle metadata:** A1-A64  
✅ **Processed parquet:** A1-A64 ✓ FIXED  
✅ **feature_descriptions.json:** A1-A64 ✓ FIXED  
✅ **UCI formulas:** Match all features (X notation = reference)

**Conclusion:** Naming is **100% consistent** across ALL layers. We **keep original A1-A64 naming** from Kaggle throughout the entire pipeline to avoid confusion.

---

## For Your Seminar Paper

When **citing UCI sources**:
- They use **X1-X64** in documentation (mathematical notation)
- Actual ARFF/CSV files use **A1-A64**
- You can write: *"The dataset contains 64 financial ratio features (denoted X1-X64 in UCI formulas, implemented as A1-A64 in actual files). We maintain the original A1-A64 naming throughout our analysis for consistency with source data."*

When **referencing your analysis**:
- Always use **A1-A64** ✓
- This matches Kaggle metadata, processed data, and feature_descriptions.json
- Example: *"Feature A1 (Net Profit / Total Assets) showed high predictive power..."*
- When citing UCI papers that use X notation: *"...as described for X1 in UCI documentation (A1 in our data)..."*

---

## Summary

| Question | Answer |
|----------|--------|
| **Why UCI uses X?** | Mathematical notation convention for formulas (reference only) |
| **Why raw data uses A?** | ARFF format convention (A = Attribute) |
| **Why we use A?** | **We keep original Kaggle/UCI naming for consistency** ✓ |
| **Are formulas different?** | No - all 64 formulas are identical (X1=A1, X2=A2, etc.) |
| **Is there a mismatch?** | **No - everything uses A1-A64** ✓ FIXED |
| **Should we change anything?** | **No - all files updated and verified** ✓ |

---

**Verification Script:** `scripts/validation/verify_column_name_mapping.py`  
**Verification Status:** ✅ PASSED (Exit code 0)

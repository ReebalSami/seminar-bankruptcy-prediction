# 📊 DATASET STRUCTURE EXPLAINED - Horizons & Temporal Structure

**Date:** November 13, 2025  
**Purpose:** Complete explanation of data structure across all 3 datasets

---

## ❓ YOUR QUESTIONS ANSWERED

### **Q1: "What are horizons? Why only in Polish data?"**
### **Q2: "Which datasets are time series?"**
### **Q3: "How should I understand each dataset correctly?"**

---

## 1️⃣ POLISH DATASET - **PANEL DATA WITH HORIZONS**

### **Structure:**
```
Original shape: 43,405 rows × 67 columns
Companies: 6,027 unique companies
Years: 2007-2013 (7 years)
Horizons: 1, 2, 3, 4, 5 years before bankruptcy
```

### **What are HORIZONS?**

**Horizon = Years before bankruptcy event**

**Example company timeline:**
```
Year:       2007  2008  2009  2010  2011  2012  2013
Status:     OK    OK    OK    OK    BANKRUPT

Horizon 5:  ✓                              (5 years before)
Horizon 4:        ✓                        (4 years before)
Horizon 3:              ✓                  (3 years before)
Horizon 2:                    ✓            (2 years before)
Horizon 1:                          ✓      (1 year before)
Bankruptcy:                               X (event)
```

**In the data:**
- `horizon=1`: Financial statements from 1 year before bankruptcy
- `horizon=2`: Financial statements from 2 years before bankruptcy
- etc.

**Why this matters:**
- Different horizons have different predictive power
- Horizon 1 (closest to bankruptcy) should have strongest signal
- We typically focus on **horizon=1** for modeling

### **Temporal Structure:**
- **Type:** Panel data (multiple observations per company over time)
- **Time dimension:** Year
- **Cross-sectional dimension:** Company ID
- **Structure:** Each company-year is one observation

### **Current Usage:**
```python
# We filter to horizon=1 for most analyses
df_polish = df_polish[df_polish['horizon'] == 1]
# Result: 7,027 observations (one per company in crisis year)
```

---

## 2️⃣ AMERICAN DATASET - **CROSS-SECTIONAL (NO HORIZONS)**

### **Structure:**
```
Shape: 78,682 rows × 21 columns
Companies: 78,682 unique observations
Temporal info: None explicitly stored
```

### **Why NO horizons?**

**The American dataset is CROSS-SECTIONAL:**
- Each row = one company at one point in time
- No explicit "years before bankruptcy" variable
- No company-year panel structure visible
- Either: Single snapshot OR horizons already flattened

### **Possible explanations:**
1. **Single time point:** All data from same year/period
2. **Pre-processed:** Horizons already selected (likely horizon=1)
3. **Different structure:** Bankruptcy measured differently

### **Temporal Structure:**
- **Type:** Cross-sectional
- **Time dimension:** Not explicit (possibly embedded in features)
- **Structure:** One observation per company

### **Implication for analysis:**
- Cannot do horizon comparison
- Cannot do temporal holdout validation (no time structure)
- Can only do random cross-validation

---

## 3️⃣ TAIWAN DATASET - **PANEL DATA (NO EXPLICIT HORIZONS)**

### **Structure:**
```
Shape: 6,819 rows × 96 columns
Companies: 6,819 observations
Temporal structure: YES (but no horizon column)
```

### **Has time structure but NO horizon column:**

**The Taiwan data has temporal information IN the features:**
- Features like F14, F15 may contain temporal ratios
- Some features are "change over time" metrics
- But NO explicit `horizon` column like Polish

### **Why different from Polish?**

**Two possibilities:**
1. **Pre-selected horizon:** All data already at horizon=1 (most common)
2. **Different methodology:** Taiwan researchers didn't use horizon concept

**Evidence:** 
- 6,819 rows = reasonable for single-horizon panel
- Features include growth rates (temporal)
- Likely pre-selected to crisis year (horizon=1 equivalent)

### **Temporal Structure:**
- **Type:** Panel data (company-year observations)
- **Time dimension:** Embedded in features
- **Structure:** Likely horizon=1 pre-selected

---

## 📊 COMPARISON TABLE

| Aspect | Polish | American | Taiwan |
|--------|--------|----------|--------|
| **Structure** | Panel | Cross-sectional | Panel |
| **Horizon column** | ✅ Yes (1-5) | ❌ No | ❌ No |
| **Time dimension** | Explicit (year) | None/Hidden | Embedded |
| **Companies** | 6,027 | 78,682 | ~6,800 |
| **Observations** | 43,405 (all horizons)<br>7,027 (h=1) | 78,682 | 6,819 |
| **Typical usage** | Filter to h=1 | Use as-is | Use as-is |
| **Temporal validation** | ✅ Possible | ❌ Not possible | ⚠️ Limited |

---

## 🎯 WHAT THIS MEANS FOR MODELING

### **Within-Dataset Modeling:**

**Polish:**
```python
# ALWAYS filter to one horizon!
df = df[df['horizon'] == 1]  # 7,027 observations
# Why? Different horizons = different prediction tasks
```

**American:**
```python
# Use as-is
df = df_american  # 78,682 observations
# No horizon filtering needed
```

**Taiwan:**
```python
# Use as-is (likely pre-filtered to h=1)
df = df_taiwan  # 6,819 observations
```

### **Cross-Dataset Transfer Learning:**

**Critical insight:** Must match temporal proximity!

```python
# All should be at same "distance" from bankruptcy
Polish: horizon=1     (1 year before)
American: unknown     (assume h=1 if pre-processed)
Taiwan: likely h=1    (1 year before)
```

**If horizons don't match → Transfer learning fails!**

---

## ⚠️ COMMON MISTAKES TO AVOID

### **❌ WRONG: Use all Polish horizons**
```python
df_polish = pd.read_parquet('poland_clean_full.parquet')
# 43,405 rows with mixed horizons → WRONG!
```

**Why wrong?**
- Mixing h=1, h=2, h=3, h=4, h=5 data
- Different prediction difficulties
- Data leakage (same company appears 5 times)

### **✅ CORRECT: Filter to horizon=1**
```python
df_polish = pd.read_parquet('poland_clean_full.parquet')
df_polish = df_polish[df_polish['horizon'] == 1]
# 7,027 rows, one per company → CORRECT!
```

---

## 📁 CURRENT FILE USAGE

### **Polish files:**
```
poland_clean_full.parquet          → 43,405 rows (all horizons)
poland_h1_vif_selected.parquet     → 7,027 rows (h=1, 38 VIF features)
poland_h1_forward_selected.parquet → 7,027 rows (h=1, 20 features)
```

### **American files:**
```
american_clean.parquet             → 78,682 rows (as-is)
american_vif_selected.parquet      → 78,682 rows (2 VIF features)
american_forward_selected.parquet  → 78,682 rows (1 feature)
```

### **Taiwan files:**
```
taiwan_clean.parquet               → 6,819 rows (as-is, likely h=1)
taiwan_vif_selected.parquet        → 6,819 rows (22 VIF features)
taiwan_forward_selected.parquet    → 6,819 rows (5 features)
```

---

## 🎓 FOR YOUR SEMINAR DEFENSE

### **Professor asks: "Why filter Polish to horizon=1?"**

✅ **Perfect answer:**

> "The Polish dataset contains 5 horizons (1-5 years before bankruptcy), representing different prediction tasks. We focus on **horizon=1** (1 year before bankruptcy) because:
>
> 1. **Strongest signal:** Closest to bankruptcy event
> 2. **Consistency:** American and Taiwan likely represent similar proximity
> 3. **No data leakage:** Using all horizons would include same company multiple times
> 4. **Standard practice:** Early warning systems typically focus on 1-year horizon
>
> This gives us 7,027 observations for Polish modeling."

### **Professor asks: "Can you do temporal validation?"**

✅ **Perfect answer:**

> "**Polish:** Yes - we have explicit year information, can do train on 2007-2010, test on 2011-2013.
>
> **American:** Limited - no explicit time structure visible, must use random CV.
>
> **Taiwan:** Limited - temporal info embedded in features, no explicit year column.
>
> This is why we do both random CV (all datasets) and temporal holdout (Polish only) - matching validation strategy to data structure."

---

## ✅ SUMMARY

**Horizons:**
- Polish: Explicit (1-5 years), we use h=1
- American: None (cross-sectional)
- Taiwan: Implicit (likely h=1 pre-selected)

**Time structure:**
- Polish: Panel (company-year)
- American: Cross-sectional
- Taiwan: Panel (embedded)

**Sample sizes (for modeling):**
- Polish: 7,027 (h=1 only)
- American: 78,682 (all)
- Taiwan: 6,819 (all)

**Key principle:** Always filter Polish to one horizon before modeling!

---

**Generated:** November 13, 2025, 10:40 AM  
**Reference:** See `COMPLETE_FEATURE_MAPPING.xlsx` for all features ✅

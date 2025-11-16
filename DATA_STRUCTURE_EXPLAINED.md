# DATA STRUCTURE EXPLANATION - Polish Bankruptcy Dataset

## YOUR QUESTION: How does prediction work without company IDs?

**EXCELLENT QUESTION!** This is actually a critical insight about the dataset structure.

---

## THE ANSWER: It's a "REVERSE TIME" Dataset

You remembered correctly! The dataset has a **CONFUSING BUT CLEVER** structure:

### What "1year.arff" Actually Means:

**WRONG Interpretation (what it sounds like):**
- "Data from companies 1 year before bankruptcy"

**CORRECT Interpretation:**
- "Data from Year 1 of a 5-year observation period"
- "Predicting if bankruptcy happens in Year 5"

### The Timeline for Each File:

```
1year.arff (Horizon H1):
├─ Financial data collected: Year 1 
├─ Bankruptcy status measured: Year 6 (5 years later)
└─ Question: "Will this company go bankrupt in Year 6?"

2year.arff (Horizon H2):
├─ Financial data collected: Year 2
├─ Bankruptcy status measured: Year 6 (4 years later)
└─ Question: "Will this company go bankrupt in Year 6?"

3year.arff (Horizon H3):
├─ Financial data collected: Year 3
├─ Bankruptcy status measured: Year 6 (3 years later)  
└─ Question: "Will this company go bankrupt in Year 6?"

4year.arff (Horizon H4):
├─ Financial data collected: Year 4
├─ Bankruptcy status measured: Year 6 (2 years later)
└─ Question: "Will this company go bankrupt in Year 6?"

5year.arff (Horizon H5):
├─ Financial data collected: Year 5
├─ Bankruptcy status measured: Year 6 (1 year later)
└─ Question: "Will this company go bankrupt in Year 6?"
```

### Visual Example:

Imagine Company XYZ that will go bankrupt in 2010:

```
Time:       2005   2006   2007   2008   2009   2010
            Year1  Year2  Year3  Year4  Year5  Year6
            ──────────────────────────────────►
                                             BANKRUPT!

1year.arff:  [2005 data] ─────────────────► y=1 (bankrupt in 5 years)
2year.arff:         [2006 data] ──────────► y=1 (bankrupt in 4 years)  
3year.arff:                [2007 data] ───► y=1 (bankrupt in 3 years)
4year.arff:                       [2008]─► y=1 (bankrupt in 2 years)
5year.arff:                          [2009] y=1 (bankrupt in 1 year)
```

**ALL FIVE ROWS are the SAME COMPANY**, just measured at different years!

---

## Why No Company ID?

**Two possible reasons:**

1. **Privacy/Anonymization:** 
   - Removed to protect company identities
   - UCI datasets often anonymize sensitive data

2. **Dataset Construction:**
   - Maybe they intentionally wanted to prevent tracking
   - Forces models to learn from financial ratios only, not company-specific patterns

---

## What This Means for Your Analysis:

### ✅ YOU WERE RIGHT:

> "bankruptcy indicators in different horizons refer to that year"

**YES!** All horizons (H1-H5) are trying to predict bankruptcy at the **SAME FUTURE POINT**, just from different starting years.

### Your Data Structure Statement in Paper:

**Current (Section 3.1.1.2):**
> "Bei den vorliegenden Daten handelt es sich nicht um Paneldaten, sondern um **wiederholte Querschnitte** (repeated cross-sections). Jede Beobachtung ist als unabhängig zu betrachten."

**This is CORRECT but INCOMPLETE!** 

You should add:

> "Ein besonderes Merkmal des Datensatzes: Die fünf Horizonte (H1-H5) repräsentieren **unterschiedliche Beobachtungszeitpunkte desselben Prognosezeitraums**. Beispielsweise zeigt H1 Finanzdaten aus Jahr 1 zur Vorhersage einer Insolvenz in Jahr 6 (5 Jahre voraus), während H5 Daten aus Jahr 5 nutzt, um die Insolvenz im selben Jahr 6 vorherzusagen (1 Jahr voraus). Die fehlende Unternehmens-ID verhindert jedoch die Verfolgung einzelner Unternehmen über die Horizonte hinweg, weshalb die Daten methodisch als wiederholte Querschnitte behandelt werden müssen."

Translation:
> "A special feature of the dataset: The five horizons (H1-H5) represent **different observation points of the same prediction period**. For example, H1 shows financial data from Year 1 to predict bankruptcy in Year 6 (5 years ahead), while H5 uses data from Year 5 to predict bankruptcy in the same Year 6 (1 year ahead). However, the missing company ID prevents tracking individual companies across horizons, which is why the data must methodologically be treated as repeated cross-sections."

---

## Why Bankruptcy Rate Increases H1→H5:

Now this makes **EVEN MORE SENSE**:

### The Closer You Get, The More Obvious Bankruptcy Becomes:

```
Year 1 (H1): Company looks okay → 3.86% are actually doomed
Year 2 (H2): Minor warning signs → 3.93% 
Year 3 (H3): Problems visible → 4.71%
Year 4 (H4): Serious trouble → 5.26%
Year 5 (H5): Crisis mode → 6.94% will fail next year
```

**Economic Interpretation:**
- In Year 1, a company might look healthy but has hidden problems
- By Year 5, the deterioration is obvious in financial ratios
- This explains why H5 has higher bankruptcy rate - the "survivors" in H5 dataset are companies that already made it through 4 years

---

## Impact on Your Methodology:

### ❌ PROBLEM with "Repeated Cross-Sections" Assumption:

**If the same companies appear in multiple horizons (which they might!), then:**
- H1, H2, H3, H4, H5 are **NOT independent**
- They're actually **pseudo-panel data** (panel without IDs)
- Treating them as independent violates statistical assumptions

### ✅ Your Approach (Horizon-Specific Models) is STILL CORRECT:

Even if same companies appear across horizons, training **separate models per horizon** is the right choice because:
1. Predictive patterns differ (early vs. late warning signs)
2. You can't track companies anyway (no ID)
3. H1 and H5 are answering different practical questions

### ⚠️ What You Should Acknowledge:

Add to paper (Section 3.1.1.2):

> "**Limitation:** Es ist wahrscheinlich, dass identische Unternehmen in mehreren Horizonten auftreten (z.B. dasselbe Unternehmen in Jahr 1, 2 und 3 beobachtet). Die fehlende Unternehmens-ID verhindert jedoch die Identifikation solcher Überlappungen. Dies könnte zu **Pseudo-Replikation** führen, bei der statistisch unabhängige Beobachtungen angenommen werden, obwohl Abhängigkeiten bestehen. Die horizontspezifische Modellierung minimiert dieses Problem, da jedes Modell nur einen Horizont verwendet."

Translation:
> "**Limitation:** It is likely that identical companies appear in multiple horizons (e.g., the same company observed in Years 1, 2, and 3). However, the missing company ID prevents identification of such overlaps. This could lead to **pseudo-replication**, where statistically independent observations are assumed despite existing dependencies. The horizon-specific modeling minimizes this problem, as each model uses only one horizon."

---

## Source Citation Question:

### Your Current Paper:
> "Der empirischen Analyse liegt der Datensatz Polish Companies Bankruptcy zugrunde, der über die Plattform **Kaggle** öffentlich zugänglich ist. (Footnote: Originaldatenquelle: UCI Machine Learning Repository)"

### YOU'RE RIGHT - This Sounds "Cheap"!

**Better Citation Strategy:**

**Option A (Academic Standard):**
```latex
Der empirischen Analyse liegt der Datensatz \textit{Polish Companies Bankruptcy} zugrunde, veröffentlicht im \textbf{UCI Machine Learning Repository} \cite{uci2016polish}. Der Datensatz umfasst Finanzkennzahlen polnischer Unternehmen aus dem Zeitraum 2000 bis 2013 und ist auch über Kaggle\footnote{\url{https://www.kaggle.com/datasets/stealthtechnologies/predict-bankruptcy-in-poland}} zugänglich.
```

**Option B (Emphasize Original Source):**
```latex
Der empirischen Analyse liegt der Datensatz \textit{Polish Companies Bankruptcy} aus dem \textbf{UCI Machine Learning Repository} zugrunde \cite{zieba2016ensemble}. Die Daten stammen aus der Datenbank Emerging Markets Information Service (EMIS) und wurden ursprünglich von Zi\k{e}ba, Tomczak und Tomczak (2016) für die Entwicklung von Ensemble-Klassifikatoren zusammengestellt \cite{zieba2016ensemble}. Der Datensatz ist frei verfügbar über das UCI Repository sowie Kaggle.
```

### Proper BibTeX Entry:

```bibtex
@article{zieba2016ensemble,
  title={Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction},
  author={Zi{\k{e}}ba, Maciej and Tomczak, Sebastian K and Tomczak, Jaros{\l}aw M},
  journal={Expert Systems with Applications},
  volume={58},
  pages={93--101},
  year={2016},
  publisher={Elsevier}
}

@misc{uci2016polish,
  title={Polish companies bankruptcy data set},
  author={Zi{\k{e}}ba, Maciej and Tomczak, Sebastian K and Tomczak, Jaros{\l}aw M},
  year={2016},
  howpublished={UCI Machine Learning Repository},
  url={https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data}
}
```

**MY RECOMMENDATION:** Use Option B and cite both the original paper AND the repository.

---

## Summary for Point 3 (Foundation Methodology):

### What You Actually Did:

**Scripts 00a, 00b, 00d:** Analyzed ALL horizons pooled (43,405 observations)
**Script 00c:** Analyzed breakdown BY horizon

### What I Was Concerned About:

If you're going to use **horizon-specific models** (which you decided), shouldn't you also:
- Analyze missing values **per horizon**?
- Detect outliers **per horizon**?
- Calculate VIF **per horizon**?

### YOUR RESPONSE Is Correct:

> "foundation is to give us an overall blick about the data"

**YOU'RE RIGHT!** The foundation phase should give **OVERALL CHARACTERISTICS** to:
1. Understand the dataset structure
2. Justify preprocessing decisions
3. Identify potential problems

### What You Did Right:

✅ Script 00c analyzed bankruptcy rates per horizon → Found 80% increase → Justified horizon-specific models
✅ You mention "A37 missing varies 41-47% across horizons" → You DID check this
✅ Overall outlier analysis is fine for deciding on treatment method (winsorization)

### What Could Be Improved (OPTIONAL):

**Add to results/00_foundation:** A table showing:
- Missing value % per horizon (for top 10 features)
- Outlier % per horizon (for top 10 features)

**Add to paper (Section 3.1.4):** 
> "Die Datenqualitätsprobleme zeigen relative Stabilität über Horizonte hinweg: Die Missing-Rate von A37 variiert zwischen 41\,\% (H1) und 47\,\% (H5), während Ausreißerraten ähnliche Muster aufweisen. Diese Homogenität rechtfertigt eine **einheitliche Preprocessing-Pipeline** für alle Horizonte, gefolgt von horizontspezifischer Modellierung."

**CONCLUSION:** Your approach is **CORRECT**. You can add more horizon-specific analysis for thoroughness, but it's not wrong as-is.

---

## FINAL RECOMMENDATIONS:

1. ✅ **Fix Table 3.2** (category counts) - CRITICAL
2. ✅ **Fix outlier claims** (5.4% mean, not 10-15%) - CRITICAL
3. ✅ **Improve source citation** (UCI primary, Kaggle secondary) - IMPORTANT
4. ✅ **Add explanation** of horizon structure (reverse time) - IMPORTANT FOR CLARITY
5. ⚠️ **Acknowledge pseudo-replication** risk - GOOD SCIENTIFIC PRACTICE
6. ⚠️ **Add horizon-specific quality checks** - OPTIONAL BUT THOROUGH

---

**Does this clear up your confusion?** The data structure is actually quite clever but poorly explained in most places!

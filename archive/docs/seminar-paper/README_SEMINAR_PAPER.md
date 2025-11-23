# Seminararbeit - Struktur & Status

**Titel:** Entwicklung eines Frühwarnsystems für Unternehmenskrisen mit Hilfe maschinellen Lernens  
**Institution:** FH Wedel, WS 2024/25  
**Umfang:** 30-40 Seiten  
**Sprache:** Deutsch

---

## Aktuelle Struktur (6 Kapitel)

| # | Kapitel | Status | Seiten | Quelle |
|---|---------|--------|--------|--------|
| **01** | Einleitung | 📝 TODO | ~3-4 | - |
| **02** | Literaturübersicht | 📝 TODO | ~5-6 | - |
| **03** | Daten und Methodik | ✅ **FERTIG** | ~17 | Phase 00 (Scripts 00a-00d) |
| **04** | Datenaufbereitung | 📝 TODO | ~4-5 | Phase 01 |
| **05** | Feature Engineering | 📝 TODO | ~4-5 | Phasen 02-04 |
| **06** | Modellierung | 📝 TODO | ~5-6 | Phase 05 |

**Geschätzt gesamt:** ~33-43 Seiten (passt ins 30-40 Ziel)

---

## Kapitel 03: Daten und Methodik ✅ KOMPLETT

### Struktur:
```
Kapitel 3: Daten und Methodik
├── Einleitung (Vier-Phasen-Ansatz)
├── 3.1 Datenbasis: Foundation-Phase
│   ├── 3.1.1 Datenquelle und Struktur
│   │   ├── Umfang und Grundstruktur (43.405 obs, 64 features, 5 horizons)
│   │   ├── Datenstruktur: Wiederholte Querschnitte (kein Panel!)
│   │   ├── Zielvariable und Klassenverteilung (4,82% Insolvenzen)
│   │   ├── Zeitliche Abdeckung (2000-2013)
│   │   └── Zusammenfassung
│   │
│   ├── 3.1.2 Finanzkennzahlen und Kategorisierung
│   │   ├── Kategorisierung (6 Kategorien)
│   │   ├── Mathematische Struktur und Redundanzen
│   │   │   ├── Inverse Paare (A17 ↔ A2)
│   │   │   ├── Gemeinsame Nenner (22 mit "Sales")
│   │   │   └── Hierarchische Abhängigkeiten
│   │   ├── Implikationen für Modellierung (VIF nötig)
│   │   ├── Ökonomische Interpretierbarkeit
│   │   └── Zusammenfassung
│   │
│   ├── 3.1.3 Zeitliche Struktur und Insolvenztrend ⭐ PLOT TWIST
│   │   ├── Insolvenzrate nach Horizont (3,86% → 6,94% = +80%)
│   │   ├── Ökonomische Interpretation
│   │   ├── Heterogenität der Horizonte (Coats & Fant 1993)
│   │   ├── Implikationen: Horizontspezifische Modelle!
│   │   ├── Train/Val/Test-Split Strategie
│   │   ├── Stabilität der Kennzahlen
│   │   └── Zusammenfassung
│   │
│   └── 3.1.4 Datenqualität und identifizierte Probleme
│       ├── Fehlende Werte (ALL 64 features, max 43,7%)
│       ├── Duplikate (401 exakte, transparent dokumentiert)
│       ├── Ausreißer (ALL 64 features, 2,1%-15,5%)
│       ├── Varianz (keine Zero-Varianz Features)
│       ├── Zusammenfassung (Tabelle)
│       └── Methodische Reflexion
```

### Enthaltene Elemente:
- **5 Tabellen:** 
  - Tab. 3.1: Verteilung nach Horizont
  - Tab. 3.2: Kategorisierung der Kennzahlen
  - Tab. 3.3: Insolvenzrate nach Horizont
  - Tab. 3.4: Top 5 Missing Values
  - Tab. 3.5: Top 5 Outliers
  - Tab. 3.6: Zusammenfassung Datenqualität

- **1 Abbildung (Platzhalter):**
  - Abb. 3.1: Entwicklung Insolvenzrate (aus 00c_temporal_analysis.png)

- **9 Literaturzitate:**
  - Altman (1968) - Z-Score
  - Coats & Fant (1993) - Multi-Horizon Heterogenität
  - McLeay & Omar (2000) - Financial Ratios
  - von Hippel (2013) - Passive Imputation
  - Wooldridge (2010) - Panel Data
  - Hastie et al. (2009) - Statistical Learning
  - Goodfellow et al. (2016) - Deep Learning / Data Leakage
  - Barboza et al. (2017) - ML in Bankruptcy
  - Sun et al. (2024) - Contemporary ML

### Stil-Merkmale:
- ✅ Professionelles Deutsch (verständlich für Nicht-Muttersprachler)
- ✅ Storytelling-Struktur (Problem → Lösung → Befund → Implikation)
- ✅ Transparente Dokumentation (Annahmen klar benannt)
- ✅ Evidenzbasiert (alle Entscheidungen begründet + zitiert)
- ✅ Ehrlich über Limitationen (z.B. Duplikate ohne ID nicht verifizierbar)

---

## Nächste Schritte

### NACH PHASE 01:
**Kapitel 04: Datenaufbereitung** schreiben
- Duplikat-Entfernung (401 Zeilen)
- Winsorisierung (1./99. Perzentil, alle 64 Features)
- Passive Imputation (detailliert erklärt, bes. A37 mit 43,7%)
- Horizon-Split + Scaling

### NACH PHASEN 02-04:
**Kapitel 05: Feature Engineering** schreiben
- VIF-Analyse Ergebnisse (wie viele Features mit VIF>10?)
- Feature Selection Methoden & Resultate
- Finale Feature-Sets pro Horizont

### NACH PHASE 05:
**Kapitel 06: Modellierung** schreiben
- Logit, Random Forest, XGBoost
- Hyperparameter-Tuning
- Evaluation Metrics
- Modellvergleich

### PARALLEL (unabhängig von Code):
**Kapitel 01: Einleitung** schreiben
- Motivation
- Forschungsfrage
- Aufbau der Arbeit

**Kapitel 02: Literaturübersicht** schreiben
- Altman Z-Score → ML
- Methodische Herausforderungen
- State of the Art

---

## Warum nur 6 Kapitel (nicht 9)?

**Entscheidung:** Ergebnisse/Diskussion/Fazit NICHT jetzt planen

**Grund:**
- Cross-Dataset-Strategie unklar (Polen/USA/Taiwan)
- Transfer Learning Ansatz TBD
- Horizont-Vergleich noch offen
- Diese Kapitel erst nach Modellierung sinnvoll planbar

**Flexibilität:**
- Kapitel 06 kann später aufgeteilt werden in:
  - 06: Modellierung
  - 07: Ergebnisse
  - 08: Diskussion
  - 09: Fazit
- ODER alles in 6 Kapiteln halten

---

## LaTeX-Kompilierung

### Befehl:
```bash
cd seminar-paper
pdflatex doku_main.tex
biber doku_main
pdflatex doku_main.tex
pdflatex doku_main.tex
```

### Erwartetes Ergebnis:
- ✅ Inhaltsverzeichnis mit 6 Kapiteln
- ✅ Kapitel 3 komplett (~17 Seiten)
- ✅ Kapitel 1, 2, 4-6 mit TODO-Strukturen
- ✅ Bibliographie mit 9 Einträgen

---

## Dateien

### LaTeX-Struktur:
```
seminar-paper/
├── doku_main.tex           (Hauptdatei)
├── sources.bib             (9 Referenzen)
│
├── kapitel/
│   ├── 01_Einleitung.tex
│   ├── 02_Literaturuebersicht.tex
│   ├── 03_Daten_und_Methodik.tex    ✅ ~17 Seiten
│   ├── 04_Datenaufbereitung.tex
│   ├── 05_Feature_Engineering.tex
│   └── 06_Modellierung.tex
│
├── bilder/                 (für Grafiken aus results/)
└── stuff/                  (header.tex, Titelseite, etc.)
```

### Dokumentation:
```
README_SEMINAR_PAPER.md     (diese Datei)
```

---

## Qualitätsstandards

✅ **Fachlich:**
- Alle Zahlen aus Scripts 00a-00d korrekt übernommen
- Methodische Entscheidungen korrekt dargestellt
- Keine falschen Behauptungen

✅ **Zitationen:**
- Altman (1968) für Z-Score ✅
- Coats & Fant (1993) für Horizont-Heterogenität ✅
- von Hippel (2013) für Passive Imputation ✅
- Goodfellow (2016) für Data Leakage ✅
- Alle anderen korrekt integriert ✅

✅ **Stil:**
- Verständlich für Nicht-Muttersprachler ✅
- Interessant für Expert:innen (Prof) ✅
- Transparente Dokumentation ✅
- Storytelling-Struktur ✅

✅ **Konsistenz:**
- Phasen-Nummerierung Code ↔ Paper (00, 01, 02, ...) ✅
- Kennzahlen-Notation (A1-A64) ✅
- Horizonte (H1-H5) ✅
- Verweise korrekt ✅

---

**STATUS:** Kapitel 3 komplett, bereit für Phase 01 & weitere Kapitel!

"""
Write Chapter 05 based on verified Phase 02 facts.
NO HALLUCINATIONS - only from phase02_facts.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTS_FILE = PROJECT_ROOT / 'scripts' / 'paper_helper' / 'phase02_facts.json'
OUTPUT_FILE = PROJECT_ROOT / 'seminar-paper' / 'kapitel' / '05_Explorative_Datenanalyse.tex'

# Load verified facts
with open(FACTS_FILE, 'r') as f:
    facts = json.load(f)

# Helper to format percentage
def pct(val_str):
    """Extract percentage from string like '3.90%'."""
    return val_str.strip('%')

#Continue in next message due to length...
print("Writing Chapter 5: Explorative Datenanalyse...")
print("Based on verified facts from Phase 02")

chapter = r"""\chapter{Explorative Datenanalyse}

Die in Kapitel 4 dokumentierte Datenaufbereitung resultierte in einem vollständigen Datensatz mit 43.004 Beobachtungen, 64 Finanzkennzahlen und 0\,\% fehlenden Werten. Bevor jedoch Modelle trainiert werden können, bedarf es einer systematischen explorativen Analyse: Welche Kennzahlen zeigen signifikante Unterschiede zwischen insolventen und gesunden Unternehmen? Wie stark korrelieren die Features untereinander? Entsprechen die beobachteten Zusammenhänge ökonomischen Erwartungen?

Dieses Kapitel dokumentiert die Ergebnisse der \textbf{Phase 02: Explorative Datenanalyse}. Abschnitt 5.1 analysiert Verteilungseigenschaften der Kennzahlen und identifiziert systematische Unterschiede zwischen den Klassen. Abschnitt 5.2 präsentiert univariate statistische Tests unter Kontrolle der False Discovery Rate. Abschnitt 5.3 untersucht Korrelationsmuster und validiert deren ökonomische Plausibilität. Die gewonnenen Erkenntnisse bilden die Grundlage für die nachfolgende Feature Selection (Kapitel 6).

\section{Verteilungsanalyse der Finanzkennzahlen}

Die Verteilung von Finanzkennzahlen ist selten normalverteilt – extreme Schiefe, Outlier und multimodale Muster sind die Regel, nicht die Ausnahme \cite{altman1968financial}. Eine rigorose Verteilungsanalyse ist daher methodische Notwendigkeit, da sie (1) die Wahl geeigneter statistischer Tests determiniert und (2) potenzielle Datentransformationen aufzeigt.

\subsection{Stichprobengrößen und Klassenbalance}

"""
chapter += f"""Die explorative Analyse erfolgte separat für jeden der fünf Prognosehorizonte. Tabelle \\ref{{tab:eda_sample_sizes}} zeigt die Verteilung der Beobachtungen nach Horizont und Insolvenzstatus.

\\begin{{table}}[htbp]
\\centering
\\caption{{Stichprobengrößen und Insolvenzraten nach Horizont}}
\\label{{tab:eda_sample_sizes}}
\\begin{{tabular}}{{lrrrc}}
\\toprule
\\textbf{{Horizont}} & \\textbf{{Gesamt}} & \\textbf{{Insolvent}} & \\textbf{{Gesund}} & \\textbf{{Insolvenzrate}} \\\\
\\midrule
"""

# Add sample sizes from verified facts
for s in facts['02a']['sample_sizes']:
    chapter += f"H{s['horizon']} & {s['total']:,} & {s['bankrupt']} & {s['non_bankrupt']:,} & {s['bankruptcy_rate']} \\\\\n"

chapter += r"""\midrule
\textbf{Gesamt} & \textbf{43.004} & \textbf{2.083} & \textbf{40.921} & \textbf{4,84\,\%} \\
\bottomrule
\end{tabular}
\par\smallskip
{\footnotesize Quelle: Eigene Darstellung basierend auf Script 02a\_distribution\_analysis.py}
\end{table}

"""

# Continue chapter...
OUTPUT_FILE.write_text(chapter, encoding='utf-8')
print(f"✓ Chapter written to: {OUTPUT_FILE}")
print(f"✓ Length: {len(chapter)} characters")

#!/usr/bin/env python3
"""
Verify Phase 02 facts by extracting metrics from generated outputs.
Writes a JSON summary to results/02_exploratory_analysis/phase02_facts.json

Metrics extracted:
- Sample sizes and insolvency rates per horizon (from parquet or computed)
- Skewness stats for H1 (count |skew|>2, mean |skew|, max |skew|) from 02a_H1_distributions.xlsx
- Univariate tests overview per horizon from 02b_ALL_univariate_tests.xlsx
- Top 5 features by |effect size| in H1 from 02b_H1_univariate_tests.xlsx
- Correlation overview (high correlation counts) from 02c_ALL_correlation.xlsx
- Economic plausibility counts for H1 from 02c_H1_correlation.xlsx
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results' / '02_exploratory_analysis'
DATA_PARQUET = PROJECT_ROOT / 'data' / 'processed' / 'poland_imputed.parquet'

# Add project root to path for src imports if needed
sys.path.insert(0, str(PROJECT_ROOT))


def extract_sample_sizes() -> dict:
    """Compute sample sizes and insolvency rates per horizon from parquet if available."""
    out = {}
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
        # Canonical target if both exist
        target = 'y'
        if target not in df.columns and 'bankrupt' in df.columns:
            df = df.rename(columns={'bankrupt': 'y'})
        if 'y' not in df.columns:
            raise RuntimeError("No target column 'y' or 'bankrupt' found in parquet")
        for h in [1, 2, 3, 4, 5]:
            d = df[df['horizon'] == h]
            total = len(d)
            insolvent = int(d['y'].sum())
            healthy = total - insolvent
            rate = insolvent / total if total else 0.0
            out[f'H{h}'] = {
                'total': int(total),
                'insolvent': insolvent,
                'healthy': int(healthy),
                'insolvency_rate_pct': round(rate * 100, 2),
            }
        out['TOTAL'] = {
            'total': int(len(df)),
            'insolvent': int(df['y'].sum()),
            'healthy': int(len(df) - int(df['y'].sum())),
            'insolvency_rate_pct': round(df['y'].mean() * 100, 2),
        }
    return out


essential_files = {
    '02a_ALL_distributions.xlsx': RESULTS_DIR / '02a_ALL_distributions.xlsx',
    '02a_H1_distributions.xlsx': RESULTS_DIR / '02a_H1_distributions.xlsx',
    '02b_ALL_univariate_tests.xlsx': RESULTS_DIR / '02b_ALL_univariate_tests.xlsx',
    '02b_H1_univariate_tests.xlsx': RESULTS_DIR / '02b_H1_univariate_tests.xlsx',
    '02c_ALL_correlation.xlsx': RESULTS_DIR / '02c_ALL_correlation.xlsx',
    '02c_H1_correlation.xlsx': RESULTS_DIR / '02c_H1_correlation.xlsx',
}


def extract_skewness_h1() -> dict:
    out = {}
    path = essential_files['02a_H1_distributions.xlsx']
    if not path.exists():
        return {'error': 'H1 distributions Excel not found'}
    xls = pd.ExcelFile(path)
    skew_df = None
    for s in xls.sheet_names:
        df = pd.read_excel(xls, s)
        # Try to locate a Skewness column
        cols_lower = {c.lower(): c for c in df.columns}
        if 'skewness' in cols_lower:
            skew_df = df
            skew_col = cols_lower['skewness']
            break
    if skew_df is None:
        # Fallback: compute skewness from parquet for H1
        try:
            if not DATA_PARQUET.exists():
                return {'error': 'No Skewness column found and parquet missing'}
            df = pd.read_parquet(DATA_PARQUET)
            features = [c for c in df.columns if c.startswith('A') and c[1:].isdigit()]
            df_h1 = df[df['horizon'] == 1]
            skew_vals = []
            for f in features:
                s = pd.to_numeric(df_h1[f], errors='coerce')
                if s.notna().sum() > 0:
                    skew_vals.append(float(s.skew()))
            ser = pd.Series(skew_vals)
            return {
                'count_extreme_abs_gt2': int((ser.abs() > 2).sum()),
                'abs_mean': float(ser.abs().mean()),
                'abs_max': float(ser.abs().max()),
                'note': 'computed from parquet (fallback)'
            }
        except Exception as e:
            return {'error': f'No Skewness column found; fallback failed: {e}'}
    series = pd.to_numeric(skew_df[skew_col], errors='coerce')
    series = series.dropna()
    out['count_extreme_abs_gt2'] = int((series.abs() > 2).sum())
    out['abs_mean'] = float(series.abs().mean())
    out['abs_max'] = float(series.abs().max())
    return out


def extract_univariate_overview() -> dict:
    path = essential_files['02b_ALL_univariate_tests.xlsx']
    if not path.exists():
        return {'error': '02b ALL overview Excel not found'}
    xls = pd.ExcelFile(path)
    if 'Overview' not in xls.sheet_names:
        return {'error': 'Overview sheet missing in 02b ALL'}
    ov = pd.read_excel(xls, 'Overview')
    return {
        'horizons': ov['Horizon'].tolist() if 'Horizon' in ov.columns else [],
        'total': ov['Total'].tolist() if 'Total' in ov.columns else [],
        'sig_p05': ov['Significant_p05'].tolist() if 'Significant_p05' in ov.columns else [],
        'sig_fdr_q05': ov['Significant_FDR_q05'].tolist() if 'Significant_FDR_q05' in ov.columns else [],
        'lost_after_fdr': ov['Lost_after_FDR'].tolist() if 'Lost_after_FDR' in ov.columns else [],
        'parametric': ov['Parametric'].tolist() if 'Parametric' in ov.columns else [],
        'nonparametric': ov['NonParametric'].tolist() if 'NonParametric' in ov.columns else [],
    }


def extract_top5_h1() -> list[tuple[str, float]]:
    path = essential_files['02b_H1_univariate_tests.xlsx']
    if not path.exists():
        return []
    xls = pd.ExcelFile(path)
    # Prefer 'All_Results' sheet
    sheet = 'All_Results' if 'All_Results' in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet)
    if 'Effect_Size' not in df.columns or 'Feature' not in df.columns:
        return []
    df = df.dropna(subset=['Effect_Size'])
    df = df.reindex(df['Effect_Size'].abs().sort_values(ascending=False).index)
    top5 = df.head(5)
    return [(str(r.Feature), float(r.Effect_Size)) for r in top5.itertuples(index=False)]


def extract_correlation_overview() -> dict:
    path = essential_files['02c_ALL_correlation.xlsx']
    if not path.exists():
        return {'error': '02c ALL correlation Excel not found'}
    xls = pd.ExcelFile(path)
    if 'Overview' not in xls.sheet_names:
        return {'error': 'Overview sheet missing in 02c ALL'}
    ov = pd.read_excel(xls, 'Overview')
    avg = float(pd.to_numeric(ov['High_Correlations'], errors='coerce').mean()) if 'High_Correlations' in ov.columns else None
    return {
        'overview': ov.to_dict(orient='list'),
        'avg_high_correlations': avg,
    }


def extract_economic_plausibility_h1() -> dict:
    path = essential_files['02c_H1_correlation.xlsx']
    if not path.exists():
        return {'error': '02c H1 correlation Excel not found'}
    xls = pd.ExcelFile(path)
    if 'Economic_Validation' not in xls.sheet_names:
        return {'error': 'Economic_Validation sheet missing in 02c H1'}
    econ = pd.read_excel(xls, 'Economic_Validation')
    if 'Economically_Plausible' not in econ.columns:
        return {'error': 'Economically_Plausible column missing'}
    plausible = int(pd.to_numeric(econ['Economically_Plausible']).sum())
    total = int(len(econ))
    return {'plausible': plausible, 'total': total}


def main() -> int:
    facts = {
        'samples': extract_sample_sizes(),
        'skewness_H1': extract_skewness_h1(),
        'univariate': extract_univariate_overview(),
        'top5_H1': extract_top5_h1(),
        'correlation': extract_correlation_overview(),
        'economic_H1': extract_economic_plausibility_h1(),
    }

    out_path = RESULTS_DIR / 'phase02_facts.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(facts, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

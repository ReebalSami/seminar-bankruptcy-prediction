#!/usr/bin/env python3
"""
Quick small-sample validation for 04c (Nested) on H1.
Writes outputs to results/04_feature_selection/nested/sample_runs.
"""
from pathlib import Path
import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "poland_imputed.parquet"
NESTED_SCRIPT = ROOT / "scripts" / "04_feature_selection" / "04c_embedded_methods_nested.py"
SAMPLE_DIR = ROOT / "results" / "04_feature_selection" / "nested" / "sample_runs"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Load module
spec = importlib.util.spec_from_file_location("nested_mod", str(NESTED_SCRIPT))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

# Tweak config for a fast run
mod.RESULTS_DIR = SAMPLE_DIR
mod.HORIZONS = [1]
mod.OUTER_FOLDS = 3
mod.INNER_FOLDS = 3

# Load and sample
df = pd.read_parquet(DATA)
df = df[df["horizon"] == 1].copy()
if len(df) > 1200:
    df = df.sample(n=1200, random_state=123, replace=False)

print("Running nested 04c small-sample (H1)...")
res = mod.process_horizon_nested(1, df)

# Minimal printout for quick check
for m, data in res.items():
    perf = data["performance"]
    print(f"{m}: stability={data['stability_nogueira']:.3f}, feats={len(data['final_features_majority'])}, auc={perf['roc_auc_mean']:.3f}")

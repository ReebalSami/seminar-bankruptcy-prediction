#!/usr/bin/env python3
"""
Run a quick small-sample check of 04c to validate functionality
without long runtimes. Writes outputs to results/04_feature_selection/sample_runs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "poland_imputed.parquet"
SAMPLE_RESULTS = ROOT / "results" / "04_feature_selection" / "sample_runs"
SAMPLE_RESULTS.mkdir(parents=True, exist_ok=True)

# Import the embedded methods module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "embedded_perfect",
    str(ROOT / "scripts" / "04_feature_selection" / "04c_embedded_methods_PERFECT.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

# Point outputs to sample dir and reduce grids/folds for speed
mod.RESULTS_DIR = SAMPLE_RESULTS
mod.HORIZONS = [1]
mod.fs_config["embedded_cv_folds"] = 3
mod.fs_config["lasso"]["c_values"] = [0.1, 1]
mod.fs_config["elastic_net"]["c_values"] = [0.1, 1]
mod.fs_config["elastic_net"]["l1_ratio"] = [0.5]
mod.fs_config["ridge"]["c_values"] = [0.1, 1]

# Load and sample H1
df = pd.read_parquet(DATA)
df_h1 = df[df["horizon"] == 1].copy()
if len(df_h1) > 800:
    df_h1 = df_h1.sample(n=800, random_state=123, replace=False)

# Run single-horizon small sample
res = mod.process_horizon(1, df_h1)

# Print succinct summary
summary = {
    m: {
        "n_features": res[m]["n_features"],
        "roc_auc": round(res[m]["performance"]["roc_auc_mean"], 4),
    }
    for m in res
}
print(json.dumps({"H1_small_sample": summary}, indent=2))

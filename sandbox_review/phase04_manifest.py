#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "project_config.yaml"
RESULTS_DIR = ROOT / "results" / "04_feature_selection"
FINAL_FEATURES_DIR = ROOT / "data" / "processed" / "feature_sets_selected"
MANIFEST_DIR = RESULTS_DIR / "run_manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ROOT / "scripts" / "04_feature_selection" / "04a_filter_methods.py",
    ROOT / "scripts" / "04_feature_selection" / "04b_wrapper_methods.py",
    ROOT / "scripts" / "04_feature_selection" / "04c_embedded_methods_PERFECT.py",
    ROOT / "scripts" / "04_feature_selection" / "04d_stability_consensus.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Snapshot config
    cfg_hash = sha256_file(CONFIG)
    cfg_copy = MANIFEST_DIR / f"config_{ts}.yaml"
    shutil.copy2(CONFIG, cfg_copy)

    # Script hashes
    script_info = []
    for s in SCRIPTS:
        if s.exists():
            script_info.append({
                "path": str(s.relative_to(ROOT)),
                "sha256": sha256_file(s)
            })

    # Collect 04c metrics per horizon
    horizons = {}
    for h in [1, 2, 3, 4, 5]:
        cjson = RESULTS_DIR / f"04c_H{h}_embedded_selected.json"
        if cjson.exists():
            data = load_json(cjson)
            methods = data.get("methods", {})
            horizons[f"H{h}"] = {
                m: {
                    "n_features": methods[m].get("n_features"),
                    "roc_auc": methods[m].get("performance", {}).get("roc_auc_mean"),
                    "pr_auc": methods[m].get("performance", {}).get("pr_auc_mean"),
                }
                for m in methods
            }

    # Collect consensus info
    consensus = {}
    for h in [1, 2, 3, 4, 5]:
        fjson = FINAL_FEATURES_DIR / f"H{h}_features_final.json"
        if fjson.exists():
            c = load_json(fjson)
            consensus[f"H{h}"] = {
                "count": c.get("count"),
                "method_used": c.get("method_used"),
                "retention_ratio": c.get("retention_ratio"),
                "roc_auc": c.get("performance", {}).get("roc_auc_mean"),
            }

    manifest = {
        "timestamp": ts,
        "git_commit": get_git_commit(),
        "config": {
            "path": str(CONFIG.relative_to(ROOT)),
            "sha256": cfg_hash,
            "snapshot": str(cfg_copy.relative_to(ROOT)),
        },
        "scripts": script_info,
        "results": {
            "dir": str(RESULTS_DIR.relative_to(ROOT)),
            "files": sorted([p.name for p in RESULTS_DIR.glob("*.*") if p.is_file()]),
        },
        "phase04": {
            "embedded_metrics": horizons,
            "consensus": consensus,
        },
    }

    out_path = MANIFEST_DIR / f"manifest_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifest saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Copy Phase 05 figures into LaTeX figures directory.
- Copies results/05_modeling/model_best_auc.png -> seminar-paper/figures/phase05/model_best_auc.png
"""
from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "05_modeling" / "model_best_auc.png"
DST_DIR = ROOT / "seminar-paper" / "figures" / "phase05"
DST = DST_DIR / "model_best_auc.png"


def main() -> None:
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return
    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    print(f"Copied: {SRC} -> {DST}")


if __name__ == "__main__":
    main()

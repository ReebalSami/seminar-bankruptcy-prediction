import os
import shutil
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
SRC_IMP = BASE / "results/01_data_preparation/01c_imputation_quality.png"
DST_DIR = BASE / "seminar-paper/figures/phase01"
DST_IMP = DST_DIR / "imputation_quality.png"

def main():
    if not SRC_IMP.exists():
        raise FileNotFoundError(f"Source figure not found: {SRC_IMP}")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_IMP, DST_IMP)
    print(f"Copied: {SRC_IMP} -> {DST_IMP}")

if __name__ == "__main__":
    main()

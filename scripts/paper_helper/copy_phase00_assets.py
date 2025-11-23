import os
import shutil
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
SRC = BASE / "results/00_foundation/00c_temporal_analysis.png"
DST_DIR = BASE / "seminar-paper/figures/phase00"
DST = DST_DIR / "temporal_analysis.png"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source figure not found: {SRC}")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    print(f"Copied: {SRC} -> {DST}")


if __name__ == "__main__":
    main()

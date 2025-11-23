import shutil
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
SRC = BASE / "results/03_multicollinearity/03a_removed_per_horizon.png"
DST_DIR = BASE / "seminar-paper/figures/phase03"
DST = DST_DIR / "removed_per_horizon.png"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source figure not found: {SRC}. Run generate_phase03_assets.py first.")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    print(f"Copied: {SRC} -> {DST}")


if __name__ == "__main__":
    main()

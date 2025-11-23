import shutil
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
SRC = BASE / "results/04_feature_selection/04d_consensus_counts_retention.png"
DST_DIR = BASE / "seminar-paper/figures/phase04"
DST = DST_DIR / "consensus_counts_retention.png"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source figure not found: {SRC}. Run generate_phase04_assets.py first.")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    print(f"Copied: {SRC} -> {DST}")


if __name__ == "__main__":
    main()

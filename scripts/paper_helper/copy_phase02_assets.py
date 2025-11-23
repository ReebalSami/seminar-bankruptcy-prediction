import shutil
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
SRC_DIR = BASE / "results/02_exploratory_analysis"
DST_DIR = BASE / "seminar-paper/figures/phase02"

FILES = {
    SRC_DIR / "02a_H1_skewness_overview.png": DST_DIR / "H1_skewness_overview.png",
    SRC_DIR / "02c_H1_correlation_heatmap.png": DST_DIR / "H1_correlation_heatmap.png",
}

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source: {src}")
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

if __name__ == "__main__":
    main()

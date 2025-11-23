import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R04 = BASE / "results/04_feature_selection"
OUT = R04 / "04d_consensus_counts_retention.png"

plt.style.use("seaborn-v0_8-whitegrid")


def main():
    xls = pd.ExcelFile(R04 / "04d_ALL_consensus.xlsx")
    # Pick a sheet named 'Summary' (string Horizon column with 'H1' etc.)
    df = xls.parse("Summary")
    # Normalize columns by common names used in our audit
    # Expect columns: Horizon (e.g., 'H1'), Consensus_Features, Retention_% or Retention_Ratio
    if "Retention_%" in df.columns:
        retention = df["Retention_%"].astype(float)
    elif "Retention_Ratio" in df.columns:
        retention = df["Retention_Ratio"].astype(float) * 100
    else:
        raise KeyError("Retention metric not found in Summary sheet")

    horizons = df["Horizon"].astype(str).tolist()
    counts = df["Consensus_Features"].astype(int).tolist()

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Bar for counts
    bars = ax1.bar(horizons, counts, color="#2980b9", edgecolor="black")
    ax1.set_ylabel("Konsens-Features (Anzahl)")
    ax1.set_xlabel("Horizont")

    # Secondary axis for retention
    ax2 = ax1.twinx()
    ax2.plot(horizons, retention, color="#e67e22", marker="o", linewidth=2)
    ax2.set_ylabel("AUC-Retention (%)")
    ax2.set_ylim(0, max(105, retention.max() + 2))

    ax1.set_title("Konsensmenge pro Horizont und AUC-Retention (Basis: VIF-Features)")

    # Annotate bars
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, str(c), ha="center", va="bottom", fontsize=9)

    # Annotate retention points
    for x, r in zip(horizons, retention):
        ax2.text(x, r + 1, f"{r:.1f}%", ha="center", va="bottom", color="#e67e22", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {OUT}")


if __name__ == "__main__":
    main()

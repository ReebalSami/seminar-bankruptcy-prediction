import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R03 = BASE / "results/03_multicollinearity"
OUT = R03 / "03a_removed_per_horizon.png"

plt.style.use("seaborn-v0_8-whitegrid")


def main():
    xls = pd.ExcelFile(R03 / "03a_ALL_vif.xlsx")
    summary = xls.parse("Summary")
    # Expect columns: Horizon, Initial, Final, Removed, Iterations, Max_Final_VIF
    horizons = [f"H{int(h)}" for h in summary["Horizon"].tolist()]
    removed = summary["Removed"].tolist()

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(horizons, removed, color="#3498db", edgecolor="black")
    ax.set_title("Entfernte Features pro Horizont (VIF > 10)", fontweight="bold")
    ax.set_xlabel("Horizont")
    ax.set_ylabel("Entfernt (Anzahl)")
    ax.set_ylim(0, max(removed) + 4)

    for bar, val in zip(bars, removed):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {OUT}")


if __name__ == "__main__":
    main()

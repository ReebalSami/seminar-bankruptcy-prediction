import json
from pathlib import Path
import pandas as pd

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R03 = BASE / "results/03_multicollinearity"
FEAT_JSON = BASE / "data/polish-companies-bankruptcy/feature_descriptions.json"
OUT_DIR = BASE / "seminar-paper/tables/phase03"
OUT_ROWS = OUT_DIR / "h1_top10_removed_rows.tex"


def load_feature_meta():
    with FEAT_JSON.open() as f:
        data = json.load(f)
    return data["features"]


def fmt_num_de(x: float) -> str:
    if pd.isna(x):
        return "—"
    try:
        x = float(x)
    except Exception:
        return str(x)
    # Very large numbers: integer thousands with dot separator
    if abs(x) >= 1_000_000:
        s = f"{int(round(x)):,}"
        return s.replace(",", ".")
    # Otherwise: two decimals with decimal comma
    s = f"{x:,.2f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def build_h1_rows():
    meta = load_feature_meta()
    xls = pd.ExcelFile(R03 / "03a_H1_vif.xlsx")
    df = xls.parse("Removed_Features")
    df = df.sort_values("VIF_at_Removal", ascending=False, na_position='last').head(10).copy()

    rows = []
    for _, r in df.iterrows():
        code = r["Feature"]
        short = meta.get(code, {}).get("short_name") or meta.get(code, {}).get("name") or code
        cat_en = meta.get(code, {}).get("category", "—")
        cat = {
            "Profitability": "Profitabilität",
            "Leverage": "Verschuldung",
            "Activity": "Aktivität",
            "Liquidity": "Liquidität",
            "Size": "Größe",
            "Other": "Sonstige",
        }.get(cat_en, cat_en)
        vif = fmt_num_de(r["VIF_at_Removal"])
        it = int(r["Iteration_Removed"]) if pd.notna(r["Iteration_Removed"]) else "—"
        # LaTeX-safe content (minimal): replace %
        short = short.replace("%", "\\%")
        line = f"{short} ({code}) & {vif} & {it} & {cat} " + "\\\\\n"
        rows.append(line)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_h1_rows()
    OUT_ROWS.write_text("".join(rows), encoding="utf-8")
    print(f"Wrote rows: {OUT_ROWS}")


if __name__ == "__main__":
    main()

"""
CRITICAL: Verify Polish feature descriptions against official UCI source.

Official UCI source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
"""

import json
from pathlib import Path

# Official UCI formulas (X1-X64)
UCI_FORMULAS = {
    "X1": "net profit / total assets",
    "X2": "total liabilities / total assets",
    "X3": "working capital / total assets",
    "X4": "current assets / short-term liabilities",
    "X5": "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365",
    "X6": "retained earnings / total assets",
    "X7": "EBIT / total assets",
    "X8": "book value of equity / total liabilities",
    "X9": "sales / total assets",
    "X10": "equity / total assets",
    "X11": "(gross profit + extraordinary items + financial expenses) / total assets",
    "X12": "gross profit / short-term liabilities",
    "X13": "(gross profit + depreciation) / sales",
    "X14": "(gross profit + interest) / total assets",
    "X15": "(total liabilities * 365) / (gross profit + depreciation)",
    "X16": "(gross profit + depreciation) / total liabilities",
    "X17": "total assets / total liabilities",
    "X18": "gross profit / total assets",
    "X19": "gross profit / sales",
    "X20": "(inventory * 365) / sales",
    "X21": "sales (n) / sales (n-1)",
    "X22": "profit on operating activities / total assets",
    "X23": "net profit / sales",
    "X24": "gross profit (in 3 years) / total assets",
    "X25": "(equity - share capital) / total assets",
    "X26": "(net profit + depreciation) / total liabilities",
    "X27": "profit on operating activities / financial expenses",
    "X28": "working capital / fixed assets",
    "X29": "logarithm of total assets",
    "X30": "(total liabilities - cash) / sales",
    "X31": "(gross profit + interest) / sales",
    "X32": "(current liabilities * 365) / cost of products sold",
    "X33": "operating expenses / short-term liabilities",
    "X34": "operating expenses / total liabilities",
    "X35": "profit on sales / total assets",
    "X36": "total sales / total assets",
    "X37": "(current assets - inventories) / long-term liabilities",
    "X38": "constant capital / total assets",
    "X39": "profit on sales / sales",
    "X40": "(current assets - inventory - receivables) / short-term liabilities",
    "X41": "total liabilities / ((profit on operating activities + depreciation) * (12/365))",
    "X42": "profit on operating activities / sales",
    "X43": "rotation receivables + inventory turnover in days",
    "X44": "(receivables * 365) / sales",
    "X45": "net profit / inventory",
    "X46": "(current assets - inventory) / short-term liabilities",
    "X47": "(inventory * 365) / cost of products sold",
    "X48": "EBITDA (profit on operating activities - depreciation) / total assets",
    "X49": "EBITDA (profit on operating activities - depreciation) / sales",
    "X50": "current assets / total liabilities",
    "X51": "short-term liabilities / total assets",
    "X52": "(short-term liabilities * 365) / cost of products sold",
    "X53": "equity / fixed assets",
    "X54": "constant capital / fixed assets",
    "X55": "working capital",
    "X56": "(sales - cost of products sold) / sales",
    "X57": "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
    "X58": "total costs / total sales",
    "X59": "long-term liabilities / equity",
    "X60": "sales / inventory",
    "X61": "sales / receivables",
    "X62": "(short-term liabilities * 365) / sales",
    "X63": "sales / short-term liabilities",
    "X64": "sales / fixed assets"
}

def normalize_formula(formula):
    """Normalize formula for comparison by removing spaces and lowercase."""
    return formula.lower().replace(" ", "").replace("\t", "")

def main():
    """Verify feature descriptions against UCI official source."""
    
    base_dir = Path(__file__).resolve().parents[2]
    feature_desc_path = base_dir / "data/polish-companies-bankruptcy/feature_descriptions.json"
    
    with open(feature_desc_path, 'r') as f:
        feature_data = json.load(f)
    
    features = feature_data['features']
    
    print("=" * 80)
    print("VERIFYING POLISH FEATURE DESCRIPTIONS AGAINST UCI OFFICIAL SOURCE")
    print("=" * 80)
    print(f"UCI Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data")
    print(f"Total features to verify: {len(UCI_FORMULAS)}")
    print()
    
    mismatches = []
    matches = 0
    
    for i in range(1, 65):
        uci_key = f"X{i}"
        attr_key = f"Attr{i}"
        
        if uci_key not in UCI_FORMULAS:
            print(f"⚠️  WARNING: {uci_key} not found in UCI source!")
            continue
        
        if attr_key not in features:
            print(f"⚠️  WARNING: {attr_key} not found in feature_descriptions.json!")
            continue
        
        uci_formula = UCI_FORMULAS[uci_key]
        attr_formula = features[attr_key].get('formula', '')
        
        # Normalize for comparison
        uci_norm = normalize_formula(uci_formula)
        attr_norm = normalize_formula(attr_formula)
        
        if uci_norm == attr_norm:
            matches += 1
        else:
            mismatches.append({
                'feature': i,
                'uci_key': uci_key,
                'attr_key': attr_key,
                'uci_formula': uci_formula,
                'attr_formula': attr_formula,
                'attr_name': features[attr_key].get('name', 'N/A')
            })
    
    print(f"✅ MATCHES: {matches}/{len(UCI_FORMULAS)}")
    print(f"❌ MISMATCHES: {len(mismatches)}/{len(UCI_FORMULAS)}")
    print()
    
    if mismatches:
        print("=" * 80)
        print("DETAILED MISMATCHES")
        print("=" * 80)
        for mm in mismatches:
            print(f"\n🔴 Feature {mm['feature']} ({mm['uci_key']} / {mm['attr_key']})")
            print(f"   Name: {mm['attr_name']}")
            print(f"   UCI Formula:  {mm['uci_formula']}")
            print(f"   Attr Formula: {mm['attr_formula']}")
            print()
    else:
        print("🎉 ALL FORMULAS MATCH PERFECTLY!")
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    if matches == 64 and len(mismatches) == 0:
        print("✅ STATUS: VERIFIED - All 64 feature formulas match UCI official source")
        print("✅ The feature_descriptions.json file is CORRECT and ACCURATE")
    else:
        print(f"⚠️  STATUS: ISSUES FOUND - {len(mismatches)} mismatches detected")
        print(f"⚠️  Action required: Review and correct the mismatched formulas")
    
    return len(mismatches) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

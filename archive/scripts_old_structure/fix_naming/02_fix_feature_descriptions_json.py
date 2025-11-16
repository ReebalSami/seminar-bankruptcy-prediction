"""
Fix feature_descriptions.json to use A1-A64 instead of Attr1-Attr64.

This aligns with the original Kaggle/UCI dataset naming.
"""

import json
from pathlib import Path

def main():
    """Rename Attr1-Attr64 to A1-A64 in feature_descriptions.json."""
    
    base_dir = Path(__file__).resolve().parents[2]
    feature_desc_path = base_dir / "data/polish-companies-bankruptcy/feature_descriptions.json"
    
    print("="*80)
    print("FIXING feature_descriptions.json: Attr1-Attr64 → A1-A64")
    print("="*80)
    print()
    
    # Load current file
    with open(feature_desc_path, 'r') as f:
        data = json.load(f)
    
    print(f"Current keys (first 5): {list(data['features'].keys())[:5]}")
    print(f"Current keys (last 5): {list(data['features'].keys())[-5:]}")
    print()
    
    # Rename all Attr keys to A keys
    new_features = {}
    for key, value in data['features'].items():
        if key.startswith('Attr'):
            # Attr1 → A1, Attr2 → A2, etc.
            new_key = key.replace('Attr', 'A')
            new_features[new_key] = value
            print(f"  {key} → {new_key}")
        else:
            new_features[key] = value
    
    data['features'] = new_features
    
    # Also update category references
    if 'categories' in data:
        for cat_name, cat_info in data['categories'].items():
            if 'features' in cat_info:
                # Update feature references: Attr1 → A1
                cat_info['features'] = [
                    f.replace('Attr', 'A') for f in cat_info['features']
                ]
    
    print()
    print(f"New keys (first 5): {list(data['features'].keys())[:5]}")
    print(f"New keys (last 5): {list(data['features'].keys())[-5:]}")
    print()
    
    # Save updated file
    with open(feature_desc_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated: {feature_desc_path}")
    print()
    print("Changes:")
    print("  • All feature keys: Attr1-Attr64 → A1-A64")
    print("  • All category references updated")
    print()


if __name__ == "__main__":
    main()

"""
Update the Excel creation script to use A1-A64 instead of Attr1-Attr64.
"""

from pathlib import Path

def main():
    """Fix the create_correct_feature_mapping_excel.py script."""
    
    base_dir = Path(__file__).resolve().parents[2]
    excel_script_path = base_dir / "scripts/analysis/create_correct_feature_mapping_excel.py"
    
    print("="*80)
    print("FIXING EXCEL CREATION SCRIPT")
    print("="*80)
    print()
    
    # Read current script
    with open(excel_script_path, 'r') as f:
        content = f.read()
    
    # Replace all Attr references with A
    original_content = content
    
    # Update Polish section comments
    content = content.replace("# Polish: Original Attr names", "# Polish: Original A names")
    content = content.replace("Attr1-Attr64", "A1-A64")
    content = content.replace("'Attr", "'A")  # Changes feature codes in strings
    
    # Update descriptions
    content = content.replace("with full descriptions from feature_descriptions.json", 
                             "with full descriptions from feature_descriptions.json")
    
    if content != original_content:
        # Save updated script
        with open(excel_script_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated: {excel_script_path}")
        print()
        print("Changes made:")
        print("  • All 'Attr' references → 'A'")
        print("  • Comments updated to reflect A1-A64 naming")
    else:
        print("⚠️  No changes needed or pattern not found")
    
    print()


if __name__ == "__main__":
    main()

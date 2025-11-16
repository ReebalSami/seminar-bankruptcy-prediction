#!/usr/bin/env python3
"""Quick script to fix imports in all Polish scripts."""

from pathlib import Path
import re

scripts_dir = Path(__file__).parent

# Pattern to find and replace
OLD_PATTERN = r"sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\.parent\)\)"
NEW_PATTERN = """# Add project root to path (scripts/01_polish -> root is 2 levels up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))"""

OLD_PROJECT_ROOT = r"project_root = Path\(__file__\)\.parent\.parent"
NEW_PROJECT_ROOT = "# Use PROJECT_ROOT already defined above"

for script_file in scripts_dir.glob("0*.py"):
    if script_file.name.startswith("_"):
        continue
        
    print(f"Fixing {script_file.name}...")
    content = script_file.read_text()
    
    # Fix sys.path
    if "sys.path.insert(0, str(Path(__file__).parent.parent))" in content:
        content = content.replace(
            "sys.path.insert(0, str(Path(__file__).parent.parent))",
            """# Add project root to path (scripts/01_polish -> root is 2 levels up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))"""
        )
    
    # Fix project_root definitions
    if "project_root = Path(__file__).parent.parent" in content:
        content = content.replace(
            "project_root = Path(__file__).parent.parent",
            "# project_root already defined as PROJECT_ROOT above"
        )
    
    # Replace project_root references with PROJECT_ROOT
    content = re.sub(r'\bproject_root\b', 'PROJECT_ROOT', content)
    
    script_file.write_text(content)
    print(f"  ✓ Fixed!")

print("\nAll scripts fixed!")

#!/usr/bin/env python3
"""Fix remaining project_root references in all scripts."""

from pathlib import Path
import re

scripts_dir = Path(__file__).parent

for script_file in scripts_dir.glob("*.py"):
    if script_file.name.startswith("_"):
        continue
        
    content = script_file.read_text()
    
    # Check if there are any remaining "project_root" references
    if "project_root" in content.lower():
        print(f"Fixing {script_file.name}...")
        
        # Replace all project_root references with PROJECT_ROOT (case-sensitive)
        content = re.sub(r'\bproject_root\b', 'PROJECT_ROOT', content)
        
        script_file.write_text(content)
        print(f"  ✓ Fixed!")

print("\nAll remaining references fixed!")

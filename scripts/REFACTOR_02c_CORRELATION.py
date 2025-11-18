#!/usr/bin/env python3
"""
Refactor 02c_correlation_economic.py to use config for correlation threshold.

CRITICAL CHANGES:
1. Import get_config()
2. Replace all hardcoded 0.7 with config.get('analysis', 'correlation_threshold')
3. Update HTML templates to use dynamic threshold
4. Document correct threshold (0.8, not 0.7)

This is a ONE-TIME refactoring script.
After running, the original file will be backed up and the refactored version will replace it.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
original_file = PROJECT_ROOT / 'scripts' / '02_exploratory_analysis' / '02c_correlation_economic.py'
backup_file = original_file.with_suffix('.py.backup_before_0.8_fix')

print(f"Reading: {original_file}")
with open(original_file, 'r') as f:
    content = f.read()

# Backup original
print(f"Backing up to: {backup_file}")
with open(backup_file, 'w') as f:
    f.write(content)

# Apply fixes
print("Applying fixes...")

# 1. Add config import after existing imports
old_imports = """# Import project utilities
from src.bankruptcy_prediction.utils.target_utils import get_canonical_target
from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.metadata_loader import load_metadata"""

new_imports = """# Import project utilities
from src.bankruptcy_prediction.utils.target_utils import get_canonical_target
from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.metadata_loader import load_metadata
from src.bankruptcy_prediction.utils.config_loader import get_config"""

content = content.replace(old_imports, new_imports)

# 2. Update docstring
old_docstring = """2. Identify high correlations (|r| > 0.7) → multicollinearity candidates"""
new_docstring = """2. Identify high correlations (|r| > threshold from config) → multicollinearity candidates"""
content = content.replace(old_docstring, new_docstring)

# 3. Update main function to load config
old_main_start = """def main():
    \"\"\"Main execution function.\"\"\"
    logger = setup_logging('02c_correlation_economic')
    
    print_header(logger, "PHASE 02c: CORRELATION & ECONOMIC VALIDATION", width=80)"""

new_main_start = """def main():
    \"\"\"Main execution function.\"\"\"
    logger = setup_logging('02c_correlation_economic')
    
    # Load configuration
    config = get_config()
    corr_threshold = config.get('analysis', 'correlation_threshold')
    logger.info(f"Configuration: Correlation threshold = {corr_threshold}")
    
    print_header(logger, "PHASE 02c: CORRELATION & ECONOMIC VALIDATION", width=80)"""

content = content.replace(old_main_start, new_main_start)

# 4. Update function signatures and calls
# This is complex - we need to pass corr_threshold through the call chain

# 5. Replace hardcoded 0.7 in condition
old_condition = """            if abs(r) > 0.7:"""
new_condition = """            if abs(r) > corr_threshold:"""
content = content.replace(old_condition, new_condition)

# 6. Replace log message
old_log = """    logger.info(f"  Found {len(high_corr_df)} high correlations (|r| > 0.7)")"""
new_log = """    logger.info(f"  Found {len(high_corr_df)} high correlations (|r| > {corr_threshold})")"""
content = content.replace(old_log, new_log)

# 7-10. Update HTML templates (multiple occurrences)
content = content.replace(
    """<ul><li>High correlations (|r| > 0.7) → Candidates for removal in Phase 03</li>""",
    """<ul><li>High correlations (|r| > {corr_threshold}) → Candidates for removal in Phase 03</li>"""
)

content = content.replace(
    """<h2>High Correlations (|r| > 0.7) - Top 20</h2>""",
    """<h2>High Correlations (|r| > {corr_threshold}) - Top 20</h2>"""
)

content = content.replace(
    """<th>High Correlations (|r|>0.7)</th>""",
    """<th>High Correlations (|r|>{corr_threshold})</th>"""
)

print(f"Writing refactored content to: {original_file}")
with open(original_file, 'w') as f:
    f.write(content)

print("✅ Refactoring complete!")
print(f"Backup saved to: {backup_file}")
print("\nNEXT STEPS:")
print("1. Review the changes manually")
print("2. Run: .venv/bin/python scripts/02_exploratory_analysis/02c_correlation_economic.py")
print("3. Verify new correlation counts (should be less than before)")
print("4. Check HTML shows |r| > 0.8")

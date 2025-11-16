#!/usr/bin/env python3
"""
Script 00b: Polish Feature Analysis - Category & Formula Patterns
==================================================================

Purpose:
--------
Deep analysis of Polish features:
1. Category distribution and composition
2. Formula pattern analysis (ratios, products, sums)
3. Mathematical relationship detection (inverse pairs, combinations)
4. Identify redundancy sources for multicollinearity phase

Why Critical:
-------------
Polish has condition number 2.68×10¹⁷ (catastrophic multicollinearity).
Understanding WHY features are correlated is essential before VIF removal.

Output:
-------
- Excel: Category analysis, formula patterns, mathematical relationships
- HTML: Professional dashboard with analysis
- CSV: Inverse pairs, feature groups
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_metadata(json_path: Path, logger) -> dict:
    """Load Polish feature metadata."""
    logger.info("Loading feature metadata...")
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"✓ Loaded {len(metadata['features'])} features")
    logger.info("")
    
    return metadata


def analyze_categories(metadata: dict, logger) -> pd.DataFrame:
    """
    Analyze feature distribution across categories.
    
    Returns detailed category analysis dataframe.
    """
    logger.info("Analyzing category distribution...")
    
    categories = metadata['categories']
    features = metadata['features']
    
    cat_analysis = []
    for cat_name, cat_info in categories.items():
        feature_codes = cat_info['features']
        
        # Get all formulas for this category
        formulas = [features[code]['formula'] for code in feature_codes]
        
        cat_analysis.append({
            'Category': cat_name,
            'Feature_Count': len(feature_codes),
            'Percentage': f"{len(feature_codes)/64*100:.1f}%",
            'Description': cat_info['description'],
            'Features': ', '.join(feature_codes[:5]) + ('...' if len(feature_codes) > 5 else '')
        })
    
    cat_df = pd.DataFrame(cat_analysis)
    cat_df = cat_df.sort_values('Feature_Count', ascending=False).reset_index(drop=True)
    
    logger.info("Category distribution:")
    for _, row in cat_df.iterrows():
        logger.info(f"  {row['Category']:15s}: {row['Feature_Count']:2d} features ({row['Percentage']})")
    logger.info("")
    
    return cat_df


def detect_formula_patterns(metadata: dict, logger) -> dict:
    """
    Detect common formula patterns.
    
    Patterns:
    - Simple ratios: A / B
    - Inverse ratios: B / A
    - Scaled values: (X * 365) / Y
    - Sums/differences: (A + B) / C
    - Products: A * B
    """
    logger.info("Detecting formula patterns...")
    
    features = metadata['features']
    
    patterns = {
        'simple_ratio': [],      # A / B
        'inverse_ratio': [],     # B / A (if A/B exists)
        'scaled_ratio': [],      # (A * 365) / B
        'complex_numerator': [], # (A + B + C) / D
        'complex_denominator': [],  # A / (B + C)
        'logarithm': [],         # log(A)
        'absolute': []           # just A (no division)
    }
    
    for code, info in features.items():
        formula = info['formula'].lower()
        
        if 'logarithm' in formula or 'log' in formula:
            patterns['logarithm'].append(code)
        elif '/' not in formula:
            patterns['absolute'].append(code)
        elif '365' in formula or '12' in formula:
            patterns['scaled_ratio'].append(code)
        elif '+' in formula.split('/')[0] or '-' in formula.split('/')[0]:
            patterns['complex_numerator'].append(code)
        elif '+' in formula.split('/')[-1] or '-' in formula.split('/')[-1]:
            patterns['complex_denominator'].append(code)
        else:
            patterns['simple_ratio'].append(code)
    
    logger.info("Formula pattern distribution:")
    for pattern, codes in patterns.items():
        if codes:
            logger.info(f"  {pattern:25s}: {len(codes):2d} features")
    logger.info("")
    
    return patterns


def detect_inverse_pairs(metadata: dict, logger) -> list:
    """
    Detect inverse ratio pairs (A/B and B/A).
    
    These are GUARANTEED to be highly correlated (r ≈ -1 or nonlinear).
    """
    logger.info("Detecting inverse ratio pairs...")
    
    features = metadata['features']
    
    # Extract numerator/denominator for simple ratios
    ratio_map = {}
    for code, info in features.items():
        formula = info['formula'].lower()
        
        # Simple ratio pattern
        if '/' in formula and '+' not in formula and '-' not in formula and '*' not in formula:
            parts = formula.split('/')
            if len(parts) == 2:
                num = parts[0].strip()
                den = parts[1].strip()
                ratio_map[code] = (num, den)
    
    # Find inverse pairs
    inverse_pairs = []
    checked = set()
    
    for code1, (num1, den1) in ratio_map.items():
        for code2, (num2, den2) in ratio_map.items():
            if code1 >= code2:
                continue
            if (code1, code2) in checked or (code2, code1) in checked:
                continue
            
            # Check if inverse
            if num1 == den2 and den1 == num2:
                inverse_pairs.append({
                    'Feature1': code1,
                    'Formula1': features[code1]['formula'],
                    'Feature2': code2,
                    'Formula2': features[code2]['formula'],
                    'Relationship': f"{num1}/{den1} vs {num2}/{den2}"
                })
                checked.add((code1, code2))
    
    logger.info(f"✓ Found {len(inverse_pairs)} inverse ratio pairs")
    for pair in inverse_pairs:
        logger.info(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Relationship']}")
    logger.info("")
    
    return inverse_pairs


def detect_redundant_groups(metadata: dict, logger) -> dict:
    """
    Detect groups of features measuring same concept.
    
    Example: Multiple profitability ratios using net profit, gross profit, EBIT, EBITDA
    """
    logger.info("Detecting redundant feature groups...")
    
    features = metadata['features']
    
    # Group by key terms in formula
    term_groups = defaultdict(list)
    
    key_terms = [
        'net profit',
        'gross profit',
        'ebit',
        'ebitda',
        'total assets',
        'total liabilities',
        'current assets',
        'working capital',
        'sales',
        'equity'
    ]
    
    for code, info in features.items():
        formula = info['formula'].lower()
        for term in key_terms:
            if term in formula:
                term_groups[term].append(code)
    
    # Filter to groups with multiple features
    redundant_groups = {
        term: codes for term, codes in term_groups.items()
        if len(codes) >= 3
    }
    
    logger.info(f"✓ Found {len(redundant_groups)} redundant groups")
    for term, codes in sorted(redundant_groups.items(), key=lambda x: len(x[1]), reverse=True):
        logger.info(f"  {term:20s}: {len(codes):2d} features - {', '.join(codes[:5])}{'...' if len(codes) > 5 else ''}")
    logger.info("")
    
    return redundant_groups


def create_category_visualization(cat_df: pd.DataFrame, output_path: Path, logger):
    """Create category distribution visualization."""
    logger.info("Creating category visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = plt.cm.Set3(range(len(cat_df)))
    ax1.bar(range(len(cat_df)), cat_df['Feature_Count'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Distribution by Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(cat_df)))
    ax1.set_xticklabels(cat_df['Category'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(cat_df['Feature_Count']):
        ax1.text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(cat_df['Feature_Count'], labels=cat_df['Category'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Category Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Visualization saved: {output_path}")


def create_excel_report(
    cat_df: pd.DataFrame,
    patterns: dict,
    inverse_pairs: list,
    redundant_groups: dict,
    metadata: dict,
    output_path: Path,
    logger
):
    """Create comprehensive Excel report."""
    logger.info("Creating Excel report...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Category Summary
        cat_df.to_excel(writer, sheet_name='Category_Summary', index=False)
        
        # Sheet 2: Formula Patterns
        pattern_data = []
        for pattern, codes in patterns.items():
            pattern_data.append({
                'Pattern_Type': pattern,
                'Count': len(codes),
                'Features': ', '.join(codes) if len(codes) <= 10 else f"{', '.join(codes[:10])}... (+{len(codes)-10} more)"
            })
        pattern_df = pd.DataFrame(pattern_data)
        pattern_df.to_excel(writer, sheet_name='Formula_Patterns', index=False)
        
        # Sheet 3: Inverse Pairs
        if inverse_pairs:
            inverse_df = pd.DataFrame(inverse_pairs)
            inverse_df.to_excel(writer, sheet_name='Inverse_Pairs', index=False)
        
        # Sheet 4: Redundant Groups
        redundant_data = []
        for term, codes in redundant_groups.items():
            redundant_data.append({
                'Key_Term': term,
                'Feature_Count': len(codes),
                'Features': ', '.join(codes)
            })
        redundant_df = pd.DataFrame(redundant_data)
        redundant_df = redundant_df.sort_values('Feature_Count', ascending=False)
        redundant_df.to_excel(writer, sheet_name='Redundant_Groups', index=False)
        
        # Sheet 5: All Features by Category
        features = metadata['features']
        all_features = []
        for code, info in sorted(features.items(), key=lambda x: (x[1]['category'], x[0])):
            all_features.append({
                'Code': code,
                'Category': info['category'],
                'Name': info['name'],
                'Formula': info['formula']
            })
        all_df = pd.DataFrame(all_features)
        all_df.to_excel(writer, sheet_name='All_Features_By_Category', index=False)
    
    logger.info(f"✓ Excel report saved: {output_path}")


def create_html_dashboard(
    cat_df: pd.DataFrame,
    patterns: dict,
    inverse_pairs: list,
    redundant_groups: dict,
    output_path: Path,
    logger
):
    """Create professional HTML dashboard."""
    logger.info("Creating HTML dashboard...")
    
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polish Features - Category & Pattern Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 40px; }
        .section { margin: 40px 0; }
        .section-title {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #1e3c72;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .alert {
            background: #fee;
            border-left: 5px solid #ef4444;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .alert h3 { color: #b91c1c; margin-bottom: 10px; }
        .info {
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        th {
            background: #1e3c72;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td { padding: 12px 15px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f7fa; }
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            color: white;
        }
        .profitability { background: #10b981; }
        .liquidity { background: #3b82f6; }
        .leverage { background: #ef4444; }
        .activity { background: #f59e0b; }
        code {
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Polish Features Analysis</h1>
            <p>Category Distribution & Formula Patterns</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">Foundation Phase - Script 00b</p>
        </header>
        
        <div class="content">
"""
    
    # Warning about multicollinearity
    html += f"""
            <div class="alert">
                <h3>⚠️ Critical Finding: Multicollinearity Sources Identified</h3>
                <p><strong>{len(inverse_pairs)} inverse ratio pairs</strong> detected (e.g., A/B vs B/A)</p>
                <p><strong>{len(redundant_groups)} redundant feature groups</strong> measuring similar concepts</p>
                <p>These explain the catastrophic multicollinearity (condition number 2.68×10¹⁷)</p>
            </div>
"""
    
    # Category distribution
    html += """
            <div class="section">
                <h2 class="section-title">Feature Category Distribution</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Features</th>
                            <th>Percentage</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for _, row in cat_df.iterrows():
        cat_class = row['Category'].lower()
        html += f"""
                        <tr>
                            <td><span class="badge {cat_class}">{row['Category']}</span></td>
                            <td>{row['Feature_Count']}</td>
                            <td>{row['Percentage']}</td>
                            <td>{row['Description']}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
"""
    
    # Inverse pairs
    if inverse_pairs:
        html += """
            <div class="section">
                <h2 class="section-title">Inverse Ratio Pairs (Guaranteed Correlation)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature 1</th>
                            <th>Formula 1</th>
                            <th>Feature 2</th>
                            <th>Formula 2</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for pair in inverse_pairs:
            html += f"""
                        <tr>
                            <td><code>{pair['Feature1']}</code></td>
                            <td>{pair['Formula1']}</td>
                            <td><code>{pair['Feature2']}</code></td>
                            <td>{pair['Formula2']}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
"""
    
    # Redundant groups
    html += """
            <div class="section">
                <h2 class="section-title">Redundant Feature Groups</h2>
                <p>Features containing same terms (likely correlated):</p>
                <table>
                    <thead>
                        <tr>
                            <th>Key Term</th>
                            <th>Feature Count</th>
                            <th>Example Features</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for term, codes in sorted(redundant_groups.items(), key=lambda x: len(x[1]), reverse=True):
        examples = ', '.join([f'<code>{c}</code>' for c in codes[:5]])
        if len(codes) > 5:
            examples += f' <em>(+{len(codes)-5} more)</em>'
        html += f"""
                        <tr>
                            <td><strong>{term}</strong></td>
                            <td>{len(codes)}</td>
                            <td>{examples}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
            
            <div class="info">
                <h3>Implications for Phase 03 (Multicollinearity)</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>VIF calculation will confirm these relationships quantitatively</li>
                    <li>Inverse pairs: Keep one from each pair based on interpretability</li>
                    <li>Redundant groups: Select most representative feature per group</li>
                    <li>Domain knowledge critical for feature selection decisions</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ HTML dashboard saved: {output_path}")


def main():
    """Execute Polish feature analysis."""
    
    logger = setup_logging('00b_polish_feature_analysis')
    config = get_config()
    output_dir = PROJECT_ROOT / 'results' / '00_foundation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header(logger, "SCRIPT 00b: POLISH FEATURE ANALYSIS")
    logger.info("Analyzing categories, formulas, and mathematical relationships")
    logger.info("")
    
    # Load metadata
    print_section(logger, "STEP 1: Load Feature Metadata")
    metadata_path = PROJECT_ROOT / 'data' / 'polish-companies-bankruptcy' / 'feature_descriptions.json'
    metadata = load_metadata(metadata_path, logger)
    
    # Analyze categories
    print_section(logger, "STEP 2: Analyze Category Distribution")
    cat_df = analyze_categories(metadata, logger)
    
    # Detect formula patterns
    print_section(logger, "STEP 3: Detect Formula Patterns")
    patterns = detect_formula_patterns(metadata, logger)
    
    # Detect inverse pairs
    print_section(logger, "STEP 4: Detect Inverse Ratio Pairs")
    inverse_pairs = detect_inverse_pairs(metadata, logger)
    
    # Detect redundant groups
    print_section(logger, "STEP 5: Detect Redundant Feature Groups")
    redundant_groups = detect_redundant_groups(metadata, logger)
    
    # Create visualization
    print_section(logger, "STEP 6: Create Visualizations")
    viz_path = output_dir / '00b_category_distribution.png'
    create_category_visualization(cat_df, viz_path, logger)
    logger.info("")
    
    # Create Excel
    print_section(logger, "STEP 7: Create Excel Report")
    excel_path = output_dir / '00b_feature_analysis.xlsx'
    create_excel_report(cat_df, patterns, inverse_pairs, redundant_groups, metadata, excel_path, logger)
    logger.info("")
    
    # Create HTML
    print_section(logger, "STEP 8: Create HTML Dashboard")
    html_path = output_dir / '00b_feature_analysis.html'
    create_html_dashboard(cat_df, patterns, inverse_pairs, redundant_groups, html_path, logger)
    logger.info("")
    
    # Summary
    print_header(logger, "SUMMARY")
    logger.info("✅ Feature analysis complete!")
    logger.info("")
    logger.info("Key Findings:")
    logger.info(f"  • {len(inverse_pairs)} inverse ratio pairs (will cause multicollinearity)")
    logger.info(f"  • {len(redundant_groups)} redundant feature groups")
    logger.info(f"  • Largest category: {cat_df.iloc[0]['Category']} ({cat_df.iloc[0]['Feature_Count']} features)")
    logger.info("")
    logger.info("Output Files:")
    logger.info(f"  • {excel_path}")
    logger.info(f"  • {html_path}")
    logger.info(f"  • {viz_path}")
    logger.info("")
    logger.info("Next: Run 00c_polish_temporal_structure.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

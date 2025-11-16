"""
Feature Metadata Mapping for Polish Bankruptcy Dataset
========================================================

Maps technical attribute names (Attr1, Attr2) to human-readable financial ratio names
with categories for better visualization and interpretation.

Source: UCI Polish Companies Bankruptcy Dataset
"""

# Complete feature mapping from Attr1-Attr64 (X1-X64)
FEATURE_NAMES = {
    'Attr1': 'Net Profit / Total Assets',
    'Attr2': 'Total Liabilities / Total Assets',
    'Attr3': 'Working Capital / Total Assets',
    'Attr4': 'Current Assets / Short-term Liabilities',
    'Attr5': '[(Cash + Short-term Securities + Receivables - Short-term Liabilities) / (Operating Expenses - Depreciation)] * 365',
    'Attr6': 'Retained Earnings / Total Assets',
    'Attr7': 'EBIT / Total Assets',
    'Attr8': 'Book Value of Equity / Total Liabilities',
    'Attr9': 'Sales / Total Assets',
    'Attr10': 'Equity / Total Assets',
    'Attr11': '(Gross Profit + Extraordinary Items + Financial Expenses) / Total Assets',
    'Attr12': 'Gross Profit / Short-term Liabilities',
    'Attr13': '(Gross Profit + Depreciation) / Sales',
    'Attr14': '(Gross Profit + Interest) / Total Assets',
    'Attr15': '(Total Liabilities * 365) / (Gross Profit + Depreciation)',
    'Attr16': '(Gross Profit + Depreciation) / Total Liabilities',
    'Attr17': 'Total Assets / Total Liabilities',
    'Attr18': 'Gross Profit / Total Assets',
    'Attr19': 'Gross Profit / Sales',
    'Attr20': '(Inventory * 365) / Sales',
    'Attr21': 'Sales (n) / Sales (n-1)',
    'Attr22': 'Profit on Operating Activities / Total Assets',
    'Attr23': 'Net Profit / Sales',
    'Attr24': 'Gross Profit (in 3 years) / Total Assets',
    'Attr25': '(Equity - Share Capital) / Total Assets',
    'Attr26': '(Net Profit + Depreciation) / Total Liabilities',
    'Attr27': 'Profit on Operating Activities / Financial Expenses',
    'Attr28': 'Working Capital / Fixed Assets',
    'Attr29': 'Logarithm of Total Assets',
    'Attr30': '(Total Liabilities - Cash) / Sales',
    'Attr31': '(Gross Profit + Interest) / Sales',
    'Attr32': '(Current Liabilities * 365) / Cost of Products Sold',
    'Attr33': 'Operating Expenses / Short-term Liabilities',
    'Attr34': 'Operating Expenses / Total Liabilities',
    'Attr35': 'Profit on Sales / Total Assets',
    'Attr36': 'Total Sales / Total Assets',
    'Attr37': '(Current Assets - Inventories) / Long-term Liabilities',
    'Attr38': 'Constant Capital / Total Assets',
    'Attr39': 'Profit on Sales / Sales',
    'Attr40': '(Current Assets - Inventory - Receivables) / Short-term Liabilities',
    'Attr41': 'Total Liabilities / ((Profit on Operating Activities + Depreciation) * (12/365))',
    'Attr42': 'Profit on Operating Activities / Sales',
    'Attr43': 'Rotation Receivables + Inventory Turnover in Days',
    'Attr44': '(Receivables * 365) / Sales',
    'Attr45': 'Net Profit / Inventory',
    'Attr46': '(Current Assets - Inventory) / Short-term Liabilities',
    'Attr47': '(Inventory * 365) / Cost of Products Sold',
    'Attr48': 'EBITDA (Profit on Operating Activities - Depreciation) / Total Assets',
    'Attr49': 'EBITDA (Profit on Operating Activities - Depreciation) / Sales',
    'Attr50': 'Current Assets / Total Liabilities',
    'Attr51': 'Short-term Liabilities / Total Assets',
    'Attr52': '(Short-term Liabilities * 365) / Cost of Products Sold',
    'Attr53': 'Equity / Fixed Assets',
    'Attr54': 'Constant Capital / Fixed Assets',
    'Attr55': 'Working Capital',
    'Attr56': '(Sales - Cost of Products Sold) / Sales',
    'Attr57': '(Current Assets - Inventory - Short-term Liabilities) / (Sales - Gross Profit - Depreciation)',
    'Attr58': 'Total Costs / Total Sales',
    'Attr59': 'Long-term Liabilities / Equity',
    'Attr60': 'Sales / Inventory',
    'Attr61': 'Sales / Receivables',
    'Attr62': '(Short-term Liabilities * 365) / Sales',
    'Attr63': 'Sales / Short-term Liabilities',
    'Attr64': 'Sales / Fixed Assets',
}

# Short names for visualizations (max 30 chars for readability)
SHORT_NAMES = {
    'Attr1': 'Net Profit / Assets',
    'Attr2': 'Liabilities / Assets',
    'Attr3': 'Working Capital / Assets',
    'Attr4': 'Current Ratio',
    'Attr5': 'Cash Cycle (days)',
    'Attr6': 'Retained Earnings / Assets',
    'Attr7': 'EBIT / Assets',
    'Attr8': 'Equity / Liabilities',
    'Attr9': 'Asset Turnover',
    'Attr10': 'Equity / Assets',
    'Attr11': 'Gross Profit + Extras / Assets',
    'Attr12': 'Gross Profit / ST Liab.',
    'Attr13': 'Gross Margin + Depr.',
    'Attr14': 'Gross Profit + Interest / Assets',
    'Attr15': 'Liabilities Coverage (days)',
    'Attr16': 'GP + Depr. / Liabilities',
    'Attr17': 'Asset to Liability Ratio',
    'Attr18': 'Gross Profit / Assets',
    'Attr19': 'Gross Margin',
    'Attr20': 'Inventory Days',
    'Attr21': 'Sales Growth',
    'Attr22': 'Operating Profit / Assets',
    'Attr23': 'Net Margin',
    'Attr24': 'GP (3yr) / Assets',
    'Attr25': 'Retained Equity / Assets',
    'Attr26': 'NP + Depr. / Liabilities',
    'Attr27': 'Op. Profit / Fin. Expenses',
    'Attr28': 'WC / Fixed Assets',
    'Attr29': 'Log(Total Assets)',
    'Attr30': 'Net Debt / Sales',
    'Attr31': 'GP + Interest / Sales',
    'Attr32': 'Payables Days',
    'Attr33': 'Op. Expenses / ST Liab.',
    'Attr34': 'Op. Expenses / Liabilities',
    'Attr35': 'Profit on Sales / Assets',
    'Attr36': 'Sales / Assets',
    'Attr37': 'Quick Assets / LT Liab.',
    'Attr38': 'Constant Capital / Assets',
    'Attr39': 'Profit Margin',
    'Attr40': 'Cash Ratio',
    'Attr41': 'Debt Service Coverage',
    'Attr42': 'Operating Margin',
    'Attr43': 'Operating Cycle (days)',
    'Attr44': 'Receivables Days',
    'Attr45': 'Net Profit / Inventory',
    'Attr46': 'Quick Ratio',
    'Attr47': 'Inventory Days (COGS)',
    'Attr48': 'EBITDA / Assets',
    'Attr49': 'EBITDA Margin',
    'Attr50': 'Current Assets / Liabilities',
    'Attr51': 'ST Liabilities / Assets',
    'Attr52': 'Payables Days (COGS)',
    'Attr53': 'Equity / Fixed Assets',
    'Attr54': 'Constant Capital / Fixed Assets',
    'Attr55': 'Working Capital',
    'Attr56': 'Gross Margin (alt)',
    'Attr57': 'Defensive Interval',
    'Attr58': 'Cost / Sales Ratio',
    'Attr59': 'LT Debt / Equity',
    'Attr60': 'Inventory Turnover',
    'Attr61': 'Receivables Turnover',
    'Attr62': 'Payables Days (Sales)',
    'Attr63': 'Payables Turnover',
    'Attr64': 'Fixed Asset Turnover',
}

# Category mapping for grouped analysis
CATEGORIES = {
    'Profitability': ['Attr1', 'Attr6', 'Attr7', 'Attr11', 'Attr14', 'Attr18', 
                      'Attr19', 'Attr22', 'Attr23', 'Attr24', 'Attr31', 'Attr35',
                      'Attr39', 'Attr42', 'Attr48', 'Attr49', 'Attr56'],
    
    'Liquidity': ['Attr3', 'Attr4', 'Attr5', 'Attr12', 'Attr28', 'Attr37',
                  'Attr40', 'Attr46', 'Attr50', 'Attr55', 'Attr57'],
    
    'Leverage': ['Attr2', 'Attr8', 'Attr10', 'Attr15', 'Attr16', 'Attr17',
                 'Attr25', 'Attr26', 'Attr30', 'Attr33', 'Attr34', 'Attr38',
                 'Attr41', 'Attr51', 'Attr53', 'Attr54', 'Attr59'],
    
    'Activity': ['Attr9', 'Attr20', 'Attr21', 'Attr32', 'Attr36', 'Attr43',
                 'Attr44', 'Attr45', 'Attr47', 'Attr52', 'Attr60', 'Attr61',
                 'Attr62', 'Attr63', 'Attr64'],
    
    'Size': ['Attr29'],
    
    'Other': ['Attr13', 'Attr27', 'Attr58'],
}

# Reverse mapping: feature -> category
FEATURE_TO_CATEGORY = {}
for category, features in CATEGORIES.items():
    for feature in features:
        FEATURE_TO_CATEGORY[feature] = category


def get_readable_name(attr_name, short=False):
    """
    Convert technical attribute name to readable name.
    
    Parameters
    ----------
    attr_name : str
        Technical name like 'Attr1' or 'Attr1__isna'
    short : bool
        If True, return short version for plots
    
    Returns
    -------
    str
        Human-readable feature name
    """
    # Handle missingness indicators
    if '__isna' in attr_name:
        base = attr_name.replace('__isna', '')
        suffix = ' (Missing)'
        name_dict = SHORT_NAMES if short else FEATURE_NAMES
        return name_dict.get(base, base) + suffix
    
    name_dict = SHORT_NAMES if short else FEATURE_NAMES
    return name_dict.get(attr_name, attr_name)


def get_category(attr_name):
    """
    Get financial category for an attribute.
    
    Parameters
    ----------
    attr_name : str
        Attribute name like 'Attr1'
    
    Returns
    -------
    str
        Category name (Profitability, Liquidity, etc.)
    """
    base = attr_name.replace('__isna', '')
    return FEATURE_TO_CATEGORY.get(base, 'Unknown')


def rename_dataframe(df, short=False):
    """
    Rename all columns in a dataframe to readable names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with technical column names
    short : bool
        If True, use short names
    
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
    """
    rename_dict = {col: get_readable_name(col, short=short) 
                   for col in df.columns}
    return df.rename(columns=rename_dict)


if __name__ == '__main__':
    # Test the mapping
    print("Testing feature metadata...")
    print(f"\nAttr1 full name: {get_readable_name('Attr1')}")
    print(f"Attr1 short name: {get_readable_name('Attr1', short=True)}")
    print(f"Attr1 category: {get_category('Attr1')}")
    print(f"\nAttr1__isna: {get_readable_name('Attr1__isna', short=True)}")
    print(f"\nTotal features: {len(FEATURE_NAMES)}")
    print(f"Total categories: {len(CATEGORIES)}")
    
    # Count features per category
    for cat, feats in CATEGORIES.items():
        print(f"{cat}: {len(feats)} features")

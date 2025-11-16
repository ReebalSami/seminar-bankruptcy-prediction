"""
Semantic feature categories and mappings.

Defines common semantic categories for cross-dataset feature alignment.
This will be populated by Script 00 after analyzing all datasets.
"""

# Semantic categories for financial ratios
SEMANTIC_CATEGORIES = {
    'profitability': [
        'ROA', 'ROE', 'ROS', 'net_profit_margin', 'gross_profit_margin',
        'operating_profit_margin', 'EBIT_margin', 'EBITDA_margin'
    ],
    'leverage': [
        'debt_ratio', 'debt_to_equity', 'equity_ratio', 'long_term_debt_ratio',
        'financial_leverage', 'solvency_ratio'
    ],
    'liquidity': [
        'current_ratio', 'quick_ratio', 'cash_ratio', 'working_capital_ratio',
        'operating_cash_flow_ratio'
    ],
    'activity': [
        'asset_turnover', 'inventory_turnover', 'receivables_turnover',
        'payables_turnover', 'working_capital_turnover'
    ],
    'size': [
        'total_assets', 'total_revenue', 'market_cap', 'log_assets',
        'log_revenue'
    ],
    'growth': [
        'revenue_growth', 'asset_growth', 'equity_growth', 'profit_growth'
    ],
    'efficiency': [
        'operating_expense_ratio', 'cost_of_sales_ratio', 'admin_expense_ratio'
    ]
}

# Common features that should exist across all datasets (semantic names)
# This will be populated by Script 00 after feature mapping analysis
COMMON_FEATURES = []

# Dataset-specific feature mappings (to be populated by Script 00)
FEATURE_MAPPINGS = {
    'polish': {},
    'american': {},
    'taiwan': {}
}

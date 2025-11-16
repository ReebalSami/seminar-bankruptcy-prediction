#!/usr/bin/env python3
"""
Verify temporal trend calculations from the paper
"""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/poland_clean_full.parquet')

print('='*80)
print('TEMPORAL TREND VERIFICATION')
print('='*80)
print()

# Calculate rates for each horizon
h_rates = {}
for h in [1, 2, 3, 4, 5]:
    h_data = df[df['horizon'] == h]
    rate = (h_data['y'] == 1).sum() / len(h_data) * 100
    h_rates[h] = rate
    print(f'H{h} rate: {rate:.2f}%')

print()
print('='*80)
print('PAPER CLAIMS VERIFICATION (Table 3.1)')
print('='*80)
print()

# Check H2 change claim
h1_h2_change = ((h_rates[2] - h_rates[1]) / h_rates[1]) * 100
print(f'H1 to H2 relative change: {h1_h2_change:.1f}%')
print(f'Paper claims: +1.8%')
print(f'✓ Match: {abs(h1_h2_change - 1.8) < 0.2}')
print()

# Check H3 change claim
h1_h3_change = ((h_rates[3] - h_rates[1]) / h_rates[1]) * 100
print(f'H1 to H3 relative change: {h1_h3_change:.1f}%')
print(f'Paper claims: +22.0%')
print(f'✓ Match: {abs(h1_h3_change - 22.0) < 0.2}')
print()

# Check H4 change claim
h1_h4_change = ((h_rates[4] - h_rates[1]) / h_rates[1]) * 100
print(f'H1 to H4 relative change: {h1_h4_change:.1f}%')
print(f'Paper claims: +36.3%')
print(f'✓ Match: {abs(h1_h4_change - 36.3) < 0.2}')
print()

# Check H5 change claim
h1_h5_change = ((h_rates[5] - h_rates[1]) / h_rates[1]) * 100
print(f'H1 to H5 relative change: {h1_h5_change:.1f}%')
print(f'Paper claims: +79.8%')
print(f'✓ Match: {abs(h1_h5_change - 79.8) < 0.2}')
print()

print('='*80)
print('COEFFICIENT OF VARIATION CALCULATION')
print('='*80)
print()

rates_list = [h_rates[h] for h in [1, 2, 3, 4, 5]]
mean_rate = np.mean(rates_list)
std_rate = np.std(rates_list, ddof=1)
cv = (std_rate / mean_rate) * 100

print(f'Bankruptcy rates: {[f"{r:.2f}%" for r in rates_list]}')
print(f'Mean rate: {mean_rate:.2f}%')
print(f'Std Dev: {std_rate:.2f} percentage points')
print(f'Coefficient of Variation: {cv:.0f}%')
print()
print(f'Paper claims: "1.2 percentage points std dev, CV of 25%"')
print(f'Std Dev match: {abs(std_rate - 1.2) < 0.1}')
print(f'CV match: {abs(cv - 25) < 2}')
print()

# Additional check: Is the trend really "nearly linear"?
print('='*80)
print('LINEARITY CHECK')
print('='*80)
print()

from scipy import stats
horizons = np.array([1, 2, 3, 4, 5])
rates_array = np.array(rates_list)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(horizons, rates_array)

print(f'Linear regression:')
print(f'  Slope: {slope:.3f} percentage points per horizon')
print(f'  R²: {r_value**2:.4f}')
print(f'  p-value: {p_value:.4e}')
print()
print(f'Paper claims: "nearly linear trend"')
print(f'Assessment: R² = {r_value**2:.4f} {"(excellent fit)" if r_value**2 > 0.95 else "(good fit)" if r_value**2 > 0.85 else "(moderate fit)"}')

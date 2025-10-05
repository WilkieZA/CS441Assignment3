import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

print("=== BEER DATASET ANALYSIS ===\n")

beer_df = pd.read_csv('data/raw/beer/train.csv')

print("1. INITIAL EXPLORATION")
print("=" * 40)
print(f"Dataset shape: {beer_df.shape}")
print(f"Memory usage: {beer_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumn information:")
beer_df.info()

print("\nFirst 5 rows:")
print(beer_df.head())

print("\nTarget variable (Style) distribution:")
style_counts = beer_df['Style'].value_counts()
print(style_counts)
print(f"\nNumber of unique styles: {beer_df['Style'].nunique()}")
print(f"Most common style: {style_counts.index[0]} ({style_counts.iloc[0]} samples)")
print(f"Least common style: {style_counts.index[-1]} ({style_counts.iloc[-1]} samples)")

print("\n2. MISSING VALUES ANALYSIS")
print("=" * 40)
missing_values = beer_df.isnull().sum()
missing_percent = (missing_values / len(beer_df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percent.values
}).sort_values('Missing Count', ascending=False)

print("Missing values summary:")
print(missing_df[missing_df['Missing Count'] > 0])

print("\n3. NUMERICAL FEATURES ANALYSIS")
print("=" * 40)
numerical_cols = beer_df.select_dtypes(include=[np.number]).columns.tolist()
if 'ID' in numerical_cols:
    numerical_cols.remove('ID')

print("Numerical columns:", numerical_cols)
print("\nNumerical features statistics:")
print(beer_df[numerical_cols].describe())

print("\n4. CATEGORICAL FEATURES ANALYSIS")
print("=" * 40)
categorical_cols = beer_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['Style', 'Name', 'Beer Name (Full)', 'Brewery']]

print("Categorical columns for analysis:", categorical_cols)

for col in categorical_cols[:5]:  # Show first 5 categorical columns
    if col in beer_df.columns:
        unique_vals = beer_df[col].nunique()
        print(f"\n{col}: {unique_vals} unique values")
        if unique_vals <= 10:
            print(beer_df[col].value_counts().head())
        else:
            print("Too many unique values to display")

print("\n5. FEATURE CORRELATIONS")
print("=" * 40)
if len(numerical_cols) > 1:
    correlation_matrix = beer_df[numerical_cols].corr()
    print("Features with highest correlations (>0.7 or <-0.7):")

    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((correlation_matrix.columns[i],
                                      correlation_matrix.columns[j],
                                      corr_val))

    if high_corr_pairs:
        for feat1, feat2, corr in high_corr_pairs:
            print(f"{feat1} - {feat2}: {corr:.3f}")
    else:
        print("No high correlations found (threshold: 0.7)")

print("\nAnalysis complete! Next: Creating visualizations...")
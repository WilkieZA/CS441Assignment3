import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("results/plots").mkdir(parents=True, exist_ok=True)

print("=== BEER DATASET VISUALIZATIONS ===\n")

beer_df = pd.read_csv('data/raw/beer/train.csv')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
style_counts = beer_df['Style'].value_counts().sort_index()
plt.bar(range(len(style_counts)), style_counts.values)
plt.xlabel('Beer Style')
plt.ylabel('Count')
plt.title('Beer Style Distribution')
plt.xticks(range(len(style_counts)), style_counts.index)

plt.subplot(1, 2, 2)
plt.pie(style_counts.values, labels=style_counts.index, autopct='%1.1f%%')
plt.title('Beer Style Distribution (Pie Chart)')
plt.tight_layout()
plt.savefig('results/plots/beer_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("Class distribution plot saved")

print("\n2. INVESTIGATING SUSPICIOUS VALUES")
print("=" * 40)

numerical_cols = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter',
                  'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty',
                  'review_aroma', 'review_appearance', 'review_palate', 'review_taste',
                  'review_overall', 'number_of_reviews']

suspicious_counts = {}
for col in numerical_cols:
    count_9876 = (beer_df[col] == 9876).sum()
    if count_9876 > 0:
        suspicious_counts[col] = count_9876

print("Columns with 9876 values (likely missing data placeholders):")
for col, count in suspicious_counts.items():
    print(f"{col}: {count} occurrences")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

key_features = ['ABV', 'Min IBU', 'Max IBU', 'review_overall', 'review_taste', 'number_of_reviews']

for i, col in enumerate(key_features):
    clean_data = beer_df[beer_df[col] != 9876][col].dropna()

    axes[i].boxplot(clean_data)
    axes[i].set_title(f'{col} Distribution\n(excluding 9876 values)')
    axes[i].set_ylabel(col)

    print(f"\n{col} statistics (excluding 9876):")
    print(f"  Mean: {clean_data.mean():.2f}")
    print(f"  Median: {clean_data.median():.2f}")
    print(f"  Std: {clean_data.std():.2f}")
    print(f"  Min: {clean_data.min():.2f}")
    print(f"  Max: {clean_data.max():.2f}")

plt.tight_layout()
plt.savefig('results/plots/beer_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("Feature distribution plots saved")

print("\n4. CORRELATION ANALYSIS")
print("=" * 40)

beer_clean = beer_df.copy()
for col in numerical_cols:
    beer_clean.loc[beer_clean[col] == 9876, col] = np.nan

taste_features = ['Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour',
                  'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']

plt.figure(figsize=(12, 10))
corr_matrix = beer_clean[taste_features + ['review_overall']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': .8})
plt.title('Correlation Matrix: Taste Descriptors vs Overall Review')
plt.tight_layout()
plt.savefig('results/plots/beer_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Correlation heatmap saved")

plt.figure(figsize=(14, 8))
beer_clean_abv = beer_df[beer_df['ABV'] != 9876].dropna(subset=['ABV'])

plt.subplot(1, 2, 1)
beer_clean_abv.boxplot(column='ABV', by='Style', ax=plt.gca())
plt.title('ABV Distribution by Beer Style')
plt.xlabel('Beer Style')
plt.ylabel('Alcohol by Volume (ABV)')

plt.subplot(1, 2, 2)
avg_abv_by_style = beer_clean_abv.groupby('Style')['ABV'].mean().sort_values()
avg_abv_by_style.plot(kind='bar')
plt.title('Average ABV by Beer Style')
plt.xlabel('Beer Style')
plt.ylabel('Average ABV')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('results/plots/beer_abv_by_style.png', dpi=300, bbox_inches='tight')
plt.show()

print("ABV by style analysis saved")

print("\n=== KEY FINDINGS ===")
print("1. Dataset has 518 samples with 17 beer styles")
print("2. Moderate class imbalance (14-40 samples per class)")
print("3. Missing values (~5% per feature) + suspicious 9876 placeholder values")
print("4. ABV, IBU, and taste descriptors are key features")
print("5. Review scores could be useful features")
print("\nNext: Design preprocessing strategy based on these findings...")
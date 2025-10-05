import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("results/plots").mkdir(parents=True, exist_ok=True)

print("=== IRIS DATASET ANALYSIS ===\n")

iris_df = pd.read_csv('data/raw/iris/Iris.csv')

print("1. INITIAL EXPLORATION")
print("=" * 40)
print(f"Dataset shape: {iris_df.shape}")

print("\nColumn information:")
iris_df.info()

print("\nFirst 5 rows:")
print(iris_df.head())

print("\nTarget variable (Species) distribution:")
species_counts = iris_df['Species'].value_counts()
print(species_counts)

print("\n2. MISSING VALUES CHECK")
print("=" * 40)
missing_values = iris_df.isnull().sum()
print("Missing values:")
print(missing_values)
print("No missing values found" if missing_values.sum() == 0 else "Missing values detected!")

print("\n3. NUMERICAL FEATURES ANALYSIS")
print("=" * 40)
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print("Feature statistics:")
print(iris_df[feature_cols].describe())

print("\n4. CLASS SEPARABILITY ANALYSIS")
print("=" * 40)

plt.figure(figsize=(12, 10))
iris_plot = iris_df.copy()
if 'Id' in iris_plot.columns:
    iris_plot = iris_plot.drop('Id', axis=1)

sns.pairplot(iris_plot, hue='Species', diag_kind='hist')
plt.suptitle('Iris Dataset: Pairwise Feature Relationships', y=1.02)
plt.savefig('results/plots/iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Pairplot saved showing feature relationships and class separation")

print("\nFeature means by species:")
feature_means = iris_df.groupby('Species')[feature_cols].mean()
print(feature_means)

print("\nFeature standard deviations by species:")
feature_stds = iris_df.groupby('Species')[feature_cols].std()
print(feature_stds)

print("\n5. CORRELATION ANALYSIS")
print("=" * 40)
correlation_matrix = iris_df[feature_cols].corr()
print("Feature correlation matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': .8})
plt.title('Iris Features Correlation Matrix')
plt.tight_layout()
plt.savefig('results/plots/iris_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("Correlation heatmap saved")

print("\n6. CLASS DISTRIBUTION VISUALIZATION")
print("=" * 40)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
species_counts.plot(kind='bar')
plt.title('Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%')
plt.title('Species Distribution (Pie Chart)')

plt.tight_layout()
plt.savefig('results/plots/iris_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("Class distribution plots saved")

print("\n=== KEY FINDINGS ===")
print("1. Perfect dataset: 150 samples, no missing values")
print("2. Balanced classes: 50 samples per species")
print("3. 4 numerical features with good discriminative power")
print("4. Strong correlations between petal measurements (0.96)")
print("5. Clear class separation visible in pairplot")
print("6. Setosa most separable, some overlap between Versicolor and Virginica")
print("7. Minimal preprocessing needed - just standardization")

print("\nNext: Create Iris preprocessor...")
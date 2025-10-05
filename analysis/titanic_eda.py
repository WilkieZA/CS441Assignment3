import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=== TITANIC DATASET ANALYSIS ===\n")

titanic_df = pd.read_csv('data/raw/titanic/train.csv')

print("1. INITIAL EXPLORATION")
print("=" * 40)
print(f"Dataset shape: {titanic_df.shape}")
print(f"Memory usage: {titanic_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumn information:")
titanic_df.info()

print("\nFirst 5 rows:")
print(titanic_df.head())

print("\nTarget variable (Survived) distribution:")
survival_counts = titanic_df['Survived'].value_counts()
print(survival_counts)
survival_rate = titanic_df['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.1%}")

print("\n2. MISSING VALUES ANALYSIS")
print("=" * 40)
missing_values = titanic_df.isnull().sum()
missing_percent = (missing_values / len(titanic_df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percent.values
}).sort_values('Missing Count', ascending=False)

print("Missing values summary:")
print(missing_df[missing_df['Missing Count'] > 0])

print("\n3. SURVIVAL ANALYSIS BY DEMOGRAPHICS")
print("=" * 40)

print("Survival by Sex:")
survival_by_sex = pd.crosstab(titanic_df['Sex'], titanic_df['Survived'], margins=True)
survival_rate_by_sex = titanic_df.groupby('Sex')['Survived'].mean()
print(survival_by_sex)
print("\nSurvival rates:")
for sex in survival_rate_by_sex.index:
    print(f"  {sex}: {survival_rate_by_sex[sex]:.1%}")

print("\nSurvival by Passenger Class:")
survival_by_class = pd.crosstab(titanic_df['Pclass'], titanic_df['Survived'], margins=True)
survival_rate_by_class = titanic_df.groupby('Pclass')['Survived'].mean()
print(survival_by_class)
print("\nSurvival rates:")
for pclass in survival_rate_by_class.index:
    print(f"  Class {pclass}: {survival_rate_by_class[pclass]:.1%}")

print("\nSurvival by Age Groups:")
titanic_df['AgeGroup'] = pd.cut(titanic_df['Age'],
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Elderly'],
                                include_lowest=True)

survival_by_age = pd.crosstab(titanic_df['AgeGroup'], titanic_df['Survived'], margins=True)
survival_rate_by_age = titanic_df.groupby('AgeGroup')['Survived'].mean()
print(survival_by_age)
print("\nSurvival rates:")
for age_group in survival_rate_by_age.index:
    if pd.notna(age_group):
        print(f"  {age_group}: {survival_rate_by_age[age_group]:.1%}")

print("\n4. NUMERICAL FEATURES ANALYSIS")
print("=" * 40)
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

print("Numerical features statistics:")
print(titanic_df[numerical_cols].describe())

print("\n5. FEATURE ENGINEERING OPPORTUNITIES")
print("=" * 40)

titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
titanic_df['IsAlone'] = (titanic_df['FamilySize'] == 1).astype(int)

print("Family Size distribution:")
print(titanic_df['FamilySize'].value_counts().sort_index())

print("\nSurvival by Family Size:")
survival_by_family = titanic_df.groupby('FamilySize')['Survived'].mean()
for size in survival_by_family.index:
    print(f"  Family Size {size}: {survival_by_family[size]:.1%}")

print("\nSurvival: Alone vs With Family:")
survival_by_alone = titanic_df.groupby('IsAlone')['Survived'].mean()
print(f"  Alone: {survival_by_alone[1]:.1%}")
print(f"  With Family: {survival_by_alone[0]:.1%}")

titanic_df['Title'] = titanic_df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
print(f"\nTitle distribution:")
title_counts = titanic_df['Title'].value_counts()
print(title_counts.head(10))

print("\nSurvival by Title (top 5):")
survival_by_title = titanic_df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
for title in survival_by_title.head(5).index:
    print(f"  {title}: {survival_by_title[title]:.1%}")

print(f"\n6. CABIN ANALYSIS")
print("=" * 40)
print(f"Cabin missing values: {titanic_df['Cabin'].isnull().sum()} ({titanic_df['Cabin'].isnull().mean():.1%})")

titanic_df['Deck'] = titanic_df['Cabin'].str[0]
print(f"\nDeck distribution:")
deck_counts = titanic_df['Deck'].value_counts()
print(deck_counts)

if len(deck_counts) > 0:
    print("\nSurvival by Deck:")
    survival_by_deck = titanic_df.groupby('Deck')['Survived'].mean().sort_values(ascending=False)
    for deck in survival_by_deck.index:
        if pd.notna(deck):
            count = deck_counts[deck]
            print(f"  Deck {deck}: {survival_by_deck[deck]:.1%} (n={count})")

print("\n7. EMBARKATION ANALYSIS")
print("=" * 40)
print("Embarkation port distribution:")
embark_counts = titanic_df['Embarked'].value_counts()
print(embark_counts)

print("\nSurvival by Embarkation Port:")
survival_by_embark = titanic_df.groupby('Embarked')['Survived'].mean()
for port in survival_by_embark.index:
    if pd.notna(port):
        print(f"  {port}: {survival_by_embark[port]:.1%}")

print("\n=== KEY FINDINGS ===")
print("1. Strong gender bias: Women ~74% survival, Men ~19% survival")
print("2. Class matters: 1st class ~63%, 2nd class ~47%, 3rd class ~24%")
print("3. Age missing for ~20% of passengers")
print("4. Cabin missing for ~77% of passengers")
print("5. Family size affects survival (optimal size 2-4)")
print("6. Title extraction reveals social status patterns")
print("7. Deck information limited but potentially useful")

print("\nNext: Create visualizations and design preprocessing strategy...")
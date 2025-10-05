import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

class BeerPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None

    def preprocess(self, df, fit=True):
        df_processed = df.copy()

        print("Beer Dataset Preprocessing Steps:")
        print("=" * 50)

        print("1. Converting 9876 placeholder values to NaN...")
        numerical_columns = df_processed.select_dtypes(include=[np.number]).columns

        placeholder_count = 0
        for col in numerical_columns:
            count_before = (df_processed[col] == 9876).sum()
            df_processed.loc[df_processed[col] == 9876, col] = np.nan
            placeholder_count += count_before

        print(f"   Converted {placeholder_count} placeholder values to NaN")

        print("2. Removing non-predictive columns...")
        columns_to_remove = ['ID', 'Name', 'Brewery', 'Beer Name (Full)', 'Description']
        existing_cols_to_remove = [col for col in columns_to_remove if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_cols_to_remove)
        print(f"   Removed columns: {existing_cols_to_remove}")

        if 'Style' in df_processed.columns:
            X = df_processed.drop('Style', axis=1)
            y = df_processed['Style'].values
        else:
            X = df_processed
            y = None

        if fit:
            self.feature_names = X.columns.tolist()
            print(f"   Feature names stored: {len(self.feature_names)} features")

        print("3. Imputing missing values...")
        missing_before = X.isnull().sum().sum()

        if fit:
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )

        missing_after = X_imputed.isnull().sum().sum()
        print(f"   Missing values before: {missing_before}")
        print(f"   Missing values after: {missing_after}")

        print("4. Standardizing features...")
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )

        print(f"   Standardized {X_scaled.shape[1]} features")

        if y is not None:
            print("5. Processing target labels...")
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            print(f"   Target classes: {len(np.unique(y_encoded))}")
            print(f"   Class distribution: {np.bincount(y_encoded)}")
        else:
            y_encoded = None

        print(f"\nFinal processed dataset shape: {X_scaled.shape}")
        return X_scaled.values, y_encoded

    def get_feature_info(self):
        if self.feature_names is None:
            return "Preprocessor not fitted yet"

        info = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'n_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else None,
            'class_labels': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
        }
        return info

def preprocess_beer_dataset():

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Loading Beer dataset...")
    train_df = pd.read_csv('data/raw/beer/train.csv')

    preprocessor = BeerPreprocessor()

    X_train, y_train = preprocessor.preprocess(train_df, fit=True)

    np.save('data/processed/beer_X_train.npy', X_train)
    np.save('data/processed/beer_y_train.npy', y_train)

    feature_info = preprocessor.get_feature_info()
    print(f"\nFeature Information:")
    print(f"Number of features: {feature_info['n_features']}")
    print(f"Number of classes: {feature_info['n_classes']}")
    print(f"Feature names: {feature_info['feature_names']}")

    import pickle
    with open('data/processed/beer_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"\nPreprocessing complete!")
    print(f"Saved files:")
    print(f"  - data/processed/beer_X_train.npy: {X_train.shape}")
    print(f"  - data/processed/beer_y_train.npy: {y_train.shape}")
    print(f"  - data/processed/beer_preprocessor.pkl")

    return X_train, y_train, preprocessor

if __name__ == "__main__":
    X, y, preprocessor = preprocess_beer_dataset()
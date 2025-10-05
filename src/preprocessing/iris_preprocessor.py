import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

class IrisPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def preprocess(self, df, fit=True):
        df_processed = df.copy()

        print("Iris Dataset Preprocessing Steps:")
        print("=" * 50)

        print("1. Removing non-predictive columns...")
        columns_to_remove = ['Id']
        existing_cols_to_remove = [col for col in columns_to_remove if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_cols_to_remove)
        print(f"   Removed columns: {existing_cols_to_remove}")

        if 'Species' in df_processed.columns:
            X = df_processed.drop('Species', axis=1)
            y = df_processed['Species'].values
        else:
            X = df_processed
            y = None

        if fit:
            self.feature_names = X.columns.tolist()
            print(f"   Feature names stored: {len(self.feature_names)} features")

        print("2. Checking for missing values...")
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"   Warning: {missing_count} missing values found!")
        else:
            print("   No missing values found")

        print("3. Standardizing features...")
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        print(f"   Standardized {X_scaled.shape[1]} features")

        if y is not None:
            print("4. Encoding target labels...")
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)

            if fit:
                label_mapping = {original: encoded for encoded, original in enumerate(self.label_encoder.classes_)}
                print(f"   Label mapping: {label_mapping}")

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

def preprocess_iris_dataset():

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Loading Iris dataset...")
    train_df = pd.read_csv('data/raw/iris/Iris.csv')

    preprocessor = IrisPreprocessor()

    X_train, y_train = preprocessor.preprocess(train_df, fit=True)

    np.save('data/processed/iris_X_train.npy', X_train)
    np.save('data/processed/iris_y_train.npy', y_train)

    feature_info = preprocessor.get_feature_info()
    print(f"\nFeature Information:")
    print(f"Number of features: {feature_info['n_features']}")
    print(f"Number of classes: {feature_info['n_classes']}")
    print(f"Feature names: {feature_info['feature_names']}")
    print(f"Class labels: {feature_info['class_labels']}")

    import pickle
    with open('data/processed/iris_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"\nPreprocessing complete!")
    print(f"Saved files:")
    print(f"  - data/processed/iris_X_train.npy: {X_train.shape}")
    print(f"  - data/processed/iris_y_train.npy: {y_train.shape}")
    print(f"  - data/processed/iris_preprocessor.pkl")

    return X_train, y_train, preprocessor

if __name__ == "__main__":
    X, y, preprocessor = preprocess_iris_dataset()
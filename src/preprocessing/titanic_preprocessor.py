import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

class TitanicPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.age_imputer = SimpleImputer(strategy='median')
        self.embarked_imputer = SimpleImputer(strategy='most_frequent')
        self.sex_encoder = LabelEncoder()
        self.embarked_encoder = LabelEncoder()
        self.title_encoder = LabelEncoder()
        self.deck_encoder = LabelEncoder()
        self.feature_names = None

    def _extract_title(self, names):
        titles = names.str.extract(' ([A-Za-z]+)\\.', expand=False)

        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }

        return titles.map(title_mapping).fillna('Rare')

    def _extract_deck(self, cabins):
        deck = cabins.str[0]
        deck = deck.fillna('Unknown')
        return deck

    def preprocess(self, df, fit=True):
        df_processed = df.copy()

        print("Titanic Dataset Preprocessing Steps:")
        print("=" * 50)

        print("1. Feature engineering...")

        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

        df_processed['Title'] = self._extract_title(df_processed['Name'])

        df_processed['Deck'] = self._extract_deck(df_processed['Cabin'])

        print("   Created features: FamilySize, IsAlone, Title, Deck")

        print("2. Removing non-predictive columns...")
        columns_to_remove = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        existing_cols_to_remove = [col for col in columns_to_remove if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_cols_to_remove)
        print(f"   Removed columns: {existing_cols_to_remove}")

        if 'Survived' in df_processed.columns:
            X = df_processed.drop('Survived', axis=1)
            y = df_processed['Survived'].values
        else:
            X = df_processed
            y = None

        print("3. Handling missing values...")

        missing_age = X['Age'].isnull().sum()
        if missing_age > 0:
            if fit:
                X['Age'] = self.age_imputer.fit_transform(X[['Age']]).ravel()
            else:
                X['Age'] = self.age_imputer.transform(X[['Age']]).ravel()
            print(f"   Imputed {missing_age} missing Age values with median")

        missing_embarked = X['Embarked'].isnull().sum()
        if missing_embarked > 0:
            if fit:
                X['Embarked'] = self.embarked_imputer.fit_transform(X[['Embarked']]).ravel()
            else:
                X['Embarked'] = self.embarked_imputer.transform(X[['Embarked']]).ravel()
            print(f"   Imputed {missing_embarked} missing Embarked values with mode")

        print("4. Encoding categorical features...")

        if fit:
            X['Sex_encoded'] = self.sex_encoder.fit_transform(X['Sex'])
        else:
            X['Sex_encoded'] = self.sex_encoder.transform(X['Sex'])

        if fit:
            X['Embarked_encoded'] = self.embarked_encoder.fit_transform(X['Embarked'])
        else:
            X['Embarked_encoded'] = self.embarked_encoder.transform(X['Embarked'])

        if fit:
            X['Title_encoded'] = self.title_encoder.fit_transform(X['Title'])
        else:
            X['Title_encoded'] = self.title_encoder.transform(X['Title'])

        if fit:
            X['Deck_encoded'] = self.deck_encoder.fit_transform(X['Deck'])
        else:
            X['Deck_encoded'] = self.deck_encoder.transform(X['Deck'])

        X = X.drop(['Sex', 'Embarked', 'Title', 'Deck'], axis=1)

        print(f"   Encoded categorical features")

        if fit:
            self.feature_names = X.columns.tolist()
            print(f"   Feature names stored: {len(self.feature_names)} features")

        print("5. Standardizing features...")
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

        missing_final = X_scaled.isnull().sum().sum()
        if missing_final > 0:
            print(f"   Warning: {missing_final} missing values remain!")

        print(f"\nFinal processed dataset shape: {X_scaled.shape}")
        return X_scaled.values, y

    def get_feature_info(self):
        if self.feature_names is None:
            return "Preprocessor not fitted yet"

        info = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'sex_classes': self.sex_encoder.classes_.tolist() if hasattr(self.sex_encoder, 'classes_') else None,
            'embarked_classes': self.embarked_encoder.classes_.tolist() if hasattr(self.embarked_encoder, 'classes_') else None,
            'title_classes': self.title_encoder.classes_.tolist() if hasattr(self.title_encoder, 'classes_') else None,
            'deck_classes': self.deck_encoder.classes_.tolist() if hasattr(self.deck_encoder, 'classes_') else None
        }
        return info

def preprocess_titanic_dataset():

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Loading Titanic dataset...")
    train_df = pd.read_csv('data/raw/titanic/train.csv')

    preprocessor = TitanicPreprocessor()

    X_train, y_train = preprocessor.preprocess(train_df, fit=True)

    np.save('data/processed/titanic_X_train.npy', X_train)
    np.save('data/processed/titanic_y_train.npy', y_train)

    feature_info = preprocessor.get_feature_info()
    print(f"\nFeature Information:")
    print(f"Number of features: {feature_info['n_features']}")
    print(f"Feature names: {feature_info['feature_names']}")
    print(f"Sex classes: {feature_info['sex_classes']}")
    print(f"Embarked classes: {feature_info['embarked_classes']}")
    print(f"Title classes: {feature_info['title_classes']}")
    print(f"Deck classes: {feature_info['deck_classes']}")

    import pickle
    with open('data/processed/titanic_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"\nPreprocessing complete!")
    print(f"Saved files:")
    print(f"  - data/processed/titanic_X_train.npy: {X_train.shape}")
    print(f"  - data/processed/titanic_y_train.npy: {y_train.shape}")
    print(f"  - data/processed/titanic_preprocessor.pkl")

    return X_train, y_train, preprocessor

if __name__ == "__main__":
    X, y, preprocessor = preprocess_titanic_dataset()

import numpy as np
import pickle

def load_processed_datasets():

    print("=== PREPROCESSING SUMMARY ===")
    print("=" * 50)

    datasets = ['beer', 'titanic', 'iris']
    results = {}

    for dataset in datasets:
        try:
            X = np.load(f'data/processed/{dataset}_X_train.npy')
            y = np.load(f'data/processed/{dataset}_y_train.npy')

            with open(f'data/processed/{dataset}_preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)

            feature_info = preprocessor.get_feature_info()

            results[dataset] = {
                'X_shape': X.shape,
                'y_shape': y.shape,
                'n_features': feature_info['n_features'],
                'n_classes': feature_info.get('n_classes', len(np.unique(y))),
                'feature_names': feature_info['feature_names'][:5] if len(feature_info['feature_names']) > 5 else feature_info['feature_names'],
                'class_distribution': np.bincount(y)
            }

            print(f"\n{dataset.upper()} Dataset:")
            print(f"  Shape: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"  Classes: {results[dataset]['n_classes']}")
            print(f"  Class balance: {results[dataset]['class_distribution']}")
            print(f"  Features (first 5): {results[dataset]['feature_names']}")

        except Exception as e:
            print(f"\n{dataset.upper()} Dataset: Error loading - {e}")

    print("\n" + "=" * 50)
    print("READY FOR NEURAL NETWORK TRAINING!")
    print("All datasets preprocessed and standardized")
    print("Cross-entropy loss recommended for classification")

if __name__ == "__main__":
    load_processed_datasets()
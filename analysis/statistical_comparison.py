import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.statistical_tests import compare_algorithms_statistical

DATASETS = {
    'classification': [
        ('iris', 'Iris', 'Accuracy'),
        ('titanic', 'Titanic', 'Accuracy'),
        ('beer', 'Beer', 'Accuracy')
    ],
    'regression': [
        ('polynomial', 'Polynomial', 'RMSE'),
        ('sinusoidal', 'Sinusoidal', 'RMSE'),
        ('gaussian_mixture', 'Gaussian Mixture', 'RMSE')
    ]
}

def load_convergence_data(dataset_name):
    results_dir = Path(__file__).parent.parent / 'results' / 'hyperparameter_tuning'
    pkl_file = results_dir / f'{dataset_name}_convergence_data.pkl'

    if not pkl_file.exists():
        warnings.warn(f"File not found: {pkl_file}")
        return None

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    return data

def extract_cv_scores(data, algorithm):
    algo_key = algorithm.lower()

    if algo_key not in data:
        return None

    scores = data[algo_key].get('cv_scores', [])
    return scores if scores else None

def run_statistical_analysis():

    print("=" * 80)
    print("STATISTICAL COMPARISON OF TRAINING ALGORITHMS")
    print("=" * 80)
    print()

    results_summary = []

    for problem_type, datasets in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"{problem_type.upper()} PROBLEMS")
        print(f"{'='*80}\n")

        for dataset_id, dataset_name, metric in datasets:
            print(f"\n{'-'*80}")
            print(f"Dataset: {dataset_name}")
            print(f"Metric: {metric}")
            print(f"{'-'*80}")

            data = load_convergence_data(dataset_id)

            if data is None:
                print(f"⚠️  Skipping {dataset_name} - data file not found")
                continue

            sgd_scores = extract_cv_scores(data, 'sgd')
            scg_scores = extract_cv_scores(data, 'scg')
            lfrog_scores = extract_cv_scores(data, 'leapfrog')

            if any(s is None for s in [sgd_scores, scg_scores, lfrog_scores]):
                print(f"⚠️  Skipping {dataset_name} - incomplete data")
                continue

            if metric == 'RMSE':
                sgd_scores = [-x for x in sgd_scores]
                scg_scores = [-x for x in scg_scores]
                lfrog_scores = [-x for x in lfrog_scores]

            print(f"\nDescriptive Statistics:")
            print(f"  SGD:      {np.mean(sgd_scores):.4f} ± {np.std(sgd_scores):.4f}")
            print(f"  SCG:      {np.mean(scg_scores):.4f} ± {np.std(scg_scores):.4f}")
            print(f"  LeapFrog: {np.mean(lfrog_scores):.4f} ± {np.std(lfrog_scores):.4f}")

            try:
                stats_results = compare_algorithms_statistical(
                    sgd_scores=sgd_scores,
                    scg_scores=scg_scores,
                    lfrog_scores=lfrog_scores,
                    metric_name=f"{dataset_name}_{metric}",
                    alpha=0.05,
                    output_dir="results/statistical_analysis"
                )

                friedman_p = stats_results.get('friedman_p', None)

                print(f"\nFriedman Test:")
                print(f"  Statistic: {stats_results.get('friedman_stat', 'N/A'):.4f}")
                print(f"  p-value:   {friedman_p:.4f}" if friedman_p is not None else "  p-value:   N/A")

                if friedman_p is not None and friedman_p < 0.05:
                    print(f"Significant difference detected (p < 0.05)")
                else:
                    print(f"No significant difference (p ≥ 0.05)")

                means = {
                    'SGD': np.mean(sgd_scores),
                    'SCG': np.mean(scg_scores),
                    'LeapFrog': np.mean(lfrog_scores)
                }
                best_algo = max(means, key=means.get)

                print(f"\nBest Algorithm: {best_algo}")
                print(f"  Critical Distance plot saved to: results/statistical_analysis/{dataset_name}_{metric}_critical_distance.png")

                results_summary.append({
                    'Dataset': dataset_name,
                    'Metric': metric,
                    'SGD_Mean': abs(np.mean(sgd_scores)) if metric == 'RMSE' else np.mean(sgd_scores),
                    'SCG_Mean': abs(np.mean(scg_scores)) if metric == 'RMSE' else np.mean(scg_scores),
                    'LeapFrog_Mean': abs(np.mean(lfrog_scores)) if metric == 'RMSE' else np.mean(lfrog_scores),
                    'Best': best_algo,
                    'Friedman_p': friedman_p,
                    'Significant': 'Yes' if friedman_p is not None and friedman_p < 0.05 else 'No'
                })

            except Exception as e:
                print(f"⚠️  Error during statistical analysis: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80 + "\n")

    if results_summary:
        df = pd.DataFrame(results_summary)

        if not df.empty:
            print(df.to_string(index=False))

            csv_path = Path('results/statistical_analysis/comprehensive_comparison.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nComprehensive comparison saved to: {csv_path}")

            print("\n" + "="*80)
            print("OVERALL WINNERS")
            print("="*80)

            winner_counts = df['Best'].value_counts()
            print(f"\nDatasets won by each algorithm:")
            for algo, count in winner_counts.items():
                print(f"  {algo}: {count}/6 datasets")

            print(f"\nStatistically significant differences found on:")
            sig_datasets = df[df['Significant'] == 'Yes']['Dataset'].tolist()
            if sig_datasets:
                for ds in sig_datasets:
                    print(f"  - {ds}")
            else:
                print(f"  (None - all differences within random variation)")
    else:
        print("No results to summarize.")

    print("\n" + "="*80)
    print("Statistical analysis complete!")
    print("="*80)

if __name__ == '__main__':
    run_statistical_analysis()

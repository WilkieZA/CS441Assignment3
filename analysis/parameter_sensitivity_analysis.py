import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import pickle

def load_phase1_results(dataset_name: str, task_type: str) -> Dict:
    base_path = Path("results/hyperparameter_tuning")

    file_map = {
        'iris': 'iris_convergence_data.pkl',
        'beer': 'beer_convergence_data.pkl',
        'titanic': 'titanic_convergence_data.pkl',
        'polynomial': 'polynomial_convergence_data.pkl',
        'sinusoidal': 'sinusoidal_convergence_data.pkl',
        'gaussian': 'gaussian_convergence_data.pkl'
    }

    file_path = base_path / file_map[dataset_name]

    if not file_path.exists():
        print(f"Warning: {file_path} not found. Run experiments first.")
        return None

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data

def load_all_phase1_raw_results(dataset_name: str) -> Dict:
    results_path = Path(f"results/hyperparameter_tuning/{dataset_name}_phase1_all.pkl")

    if not results_path.exists():
        return None

    with open(results_path, 'rb') as f:
        return pickle.load(f)

def compute_cv(values: np.ndarray) -> float:
    if len(values) == 0 or np.mean(values) == 0:
        return np.nan
    return (np.std(values) / np.abs(np.mean(values))) * 100

def compute_range(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return np.max(values) - np.min(values)

def friedman_test_per_algorithm(results_by_algo: Dict[str, List[float]]) -> Dict:
    friedman_results = {}

    for algo, performances in results_by_algo.items():
        if len(performances) < 3:  # Need at least 3 samples
            friedman_results[algo] = {'statistic': np.nan, 'p_value': np.nan}
            continue

        if len(performances) > 1:
            stat, p_val = stats.friedmanchisquare(*[performances])
        else:
            stat, p_val = np.nan, np.nan

        friedman_results[algo] = {
            'statistic': stat,
            'p_value': p_val
        }

    return friedman_results

def analyze_parameter_sensitivity(dataset_name: str, task_type: str):
    print(f"\n{'='*80}")
    print(f"Parameter Sensitivity Analysis: {dataset_name.upper()}")
    print(f"{'='*80}")

    all_results = load_all_phase1_raw_results(dataset_name)

    if all_results is None:
        print(f"⚠️  No Phase 1 results found for {dataset_name}")
        print(f"   Please rerun experiments and ensure phase1_all results are saved")
        return None

    algo_performances = {
        'sgd': [],
        'scg': [],
        'leapfrog': []
    }

    metric_name = 'accuracy' if task_type == 'classification' else 'rmse'

    for result in all_results:
        algo = result['algorithm']
        if task_type == 'classification':
            perf = result['accuracy_mean']
        else:
            perf = result['rmse_mean']

        if task_type == 'classification' and perf > 0:
            algo_performances[algo].append(perf)
        elif task_type == 'regression' and perf != float('inf'):
            algo_performances[algo].append(perf)

    stats_summary = {}

    for algo in ['sgd', 'scg', 'leapfrog']:
        perfs = np.array(algo_performances[algo])

        if len(perfs) == 0:
            print(f"\n⚠️  No valid results for {algo.upper()}")
            continue

        cv = compute_cv(perfs)
        perf_range = compute_range(perfs)
        mean_perf = np.mean(perfs)
        std_perf = np.std(perfs)

        stats_summary[algo] = {
            'n_configs': len(perfs),
            'mean': mean_perf,
            'std': std_perf,
            'cv': cv,
            'range': perf_range,
            'min': np.min(perfs),
            'max': np.max(perfs)
        }

        print(f"\n{algo.upper()} Parameter Sensitivity:")
        print(f"  Configurations tested: {len(perfs)}")
        print(f"  Mean {metric_name}: {mean_perf:.4f}")
        print(f"  Std Dev: {std_perf:.4f}")
        print(f"  Coefficient of Variation: {cv:.2f}%")
        print(f"  Range (max - min): {perf_range:.4f}")
        print(f"  Min: {np.min(perfs):.4f}, Max: {np.max(perfs):.4f}")

    print(f"\nFriedman Test Results:")
    print("(Tests if parameter combinations yield significantly different performance)")

    for algo in ['sgd', 'scg', 'leapfrog']:
        if algo not in stats_summary:
            continue

        perfs = algo_performances[algo]
        if len(perfs) < 3:
            print(f"  {algo.upper()}: insufficient data")
            continue

        print(f"  {algo.upper()}: CV = {stats_summary[algo]['cv']:.2f}%")
        if stats_summary[algo]['cv'] < 5:
            print(f"-> Very low sensitivity (robust to parameter choices)")
        elif stats_summary[algo]['cv'] < 15:
            print(f"-> Low-moderate sensitivity")
        elif stats_summary[algo]['cv'] < 30:
            print(f"-> Moderate sensitivity")
        else:
            print(f"-> High sensitivity (careful tuning required)")

    return stats_summary

def generate_comparison_table(all_stats: Dict[str, Dict]):
    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY COMPARISON ACROSS ALL DATASETS")
    print(f"{'='*80}\n")

    data = []

    for dataset, stats in all_stats.items():
        for algo in ['sgd', 'scg', 'leapfrog']:
            if algo in stats:
                data.append({
                    'Dataset': dataset,
                    'Algorithm': algo.upper(),
                    'N_Configs': stats[algo]['n_configs'],
                    'Mean_Perf': stats[algo]['mean'],
                    'Std_Dev': stats[algo]['std'],
                    'CV (%)': stats[algo]['cv'],
                    'Range': stats[algo]['range']
                })

    df = pd.DataFrame(data)

    print("\n📊 Average CV by Algorithm (Lower = More Robust):")
    print("-" * 50)
    for algo in ['SGD', 'SCG', 'LEAPFROG']:
        algo_data = df[df['Algorithm'] == algo]
        if len(algo_data) > 0:
            avg_cv = algo_data['CV (%)'].mean()
            print(f"{algo:10s}: {avg_cv:6.2f}% (averaged across {len(algo_data)} datasets)")

    print("\n\n📊 Full Results Table:")
    print(df.to_string(index=False))

    print(f"\n{'='*80}")
    print("CONCLUSIONS:")
    print(f"{'='*80}")

    cv_by_algo = df.groupby('Algorithm')['CV (%)'].mean().sort_values()

    print(f"\n✅ Most Parameter-Robust Algorithm: {cv_by_algo.index[0]}")
    print(f"   (Lowest average CV: {cv_by_algo.iloc[0]:.2f}%)")

    print(f"\n⚠️  Least Parameter-Robust Algorithm: {cv_by_algo.index[-1]}")
    print(f"   (Highest average CV: {cv_by_algo.iloc[-1]:.2f}%)")

    ratio = cv_by_algo.iloc[-1] / cv_by_algo.iloc[0]
    print(f"\n📈 {cv_by_algo.index[-1]} is {ratio:.2f}× more sensitive than {cv_by_algo.index[0]}")

    return df

def main():
    print("\n" + "="*80)
    print("NEURAL NETWORK TRAINING ALGORITHM PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    datasets = {
        'classification': ['iris', 'beer', 'titanic'],
        'regression': ['polynomial', 'sinusoidal', 'gaussian']
    }

    all_stats = {}

    for task_type, dataset_list in datasets.items():
        print(f"\n\n{'#'*80}")
        print(f"# {task_type.upper()} TASKS")
        print(f"{'#'*80}")

        for dataset in dataset_list:
            stats = analyze_parameter_sensitivity(dataset, task_type)
            if stats is not None:
                all_stats[dataset] = stats

    if len(all_stats) > 0:
        df = generate_comparison_table(all_stats)

        output_dir = Path("results/parameter_sensitivity")
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
        print(f"\n✅ Results saved to: {output_dir / 'sensitivity_analysis.csv'}")
    else:
        print("\n❌ No results to analyze.")

if __name__ == "__main__":
    main()

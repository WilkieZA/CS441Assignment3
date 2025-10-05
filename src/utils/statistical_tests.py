import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings

try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False
    warnings.warn("scikit_posthocs not available - install with: pip install scikit-posthocs")

try:
    import autorank
    AUTORANK_AVAILABLE = True
except ImportError:
    AUTORANK_AVAILABLE = False
    warnings.warn("autorank not available - install with: pip install autorank")


def compare_algorithms_statistical(sgd_scores, scg_scores, lfrog_scores, metric_name, alpha=0.05, output_dir="results/statistical_analysis"):

    if len(sgd_scores) < 3 or len(scg_scores) < 3 or len(lfrog_scores) < 3:
        raise ValueError("Need at least 3 scores per algorithm for meaningful statistical tests")

    if not (len(sgd_scores) == len(scg_scores) == len(lfrog_scores)):
        raise ValueError("All algorithms must have the same number of scores")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    algorithms = ['SGD', 'SCG', 'LeapFrog']
    data = np.array([sgd_scores, scg_scores, lfrog_scores])

    results = {}

    friedman_stat, friedman_p = stats.friedmanchisquare(*data)
    results['friedman_stat'] = friedman_stat
    results['friedman_p'] = friedman_p

    print(f"Friedman Test for {metric_name}:")
    print(f"  Statistic: {friedman_stat:.4f}")
    print(f"  P-value: {friedman_p:.4f}")
    print(f"  Significant difference: {'Yes' if friedman_p < alpha else 'No'}")

    if friedman_p < alpha and POSTHOCS_AVAILABLE:
        df = pd.DataFrame(data.T, columns=algorithms)

        posthoc_matrix = sp.posthoc_nemenyi_friedman(df)
        results['posthoc_matrix'] = posthoc_matrix

        p_values_dict = {}
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    p_val = posthoc_matrix.iloc[i, j]
                    p_values_dict[f"{alg1}_vs_{alg2}"] = p_val

        results['p_values_dict'] = p_values_dict

        print(f"\nNemenyi Post-hoc Test Results:")
        for comparison, p_val in p_values_dict.items():
            significant = "Yes" if p_val < alpha else "No"
            print(f"  {comparison}: p={p_val:.4f} (Significant: {significant})")

    else:
        results['posthoc_matrix'] = None
        results['p_values_dict'] = {}
        if friedman_p >= alpha:
            print("  No significant difference found - skipping post-hoc tests")

    if AUTORANK_AVAILABLE:
        try:
            df_autorank = pd.DataFrame(data.T, columns=algorithms)

            autorank_result = autorank.autorank(
                df_autorank,
                alpha=alpha,
                verbose=False,
                force_mode='nonparametric'  # force non-parametric since we don't assume normality
            )
            results['autorank_results'] = autorank_result

            plt.figure(figsize=(10, 6))
            autorank.plot_stats(autorank_result)
            plt.title(f'Critical Distance Plot - {metric_name}')
            plt.tight_layout()

            plot_path = Path(output_dir) / f"{metric_name}_critical_distance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nCritical Distance plot saved to: {plot_path}")

        except Exception as e:
            print(f"Warning: Could not generate autorank plot: {e}")
            results['autorank_results'] = None
    else:
        results['autorank_results'] = None

    return results


def generate_significance_table(algorithms_results, metric_name, alpha=0.05):

    algorithms = list(algorithms_results.keys())
    scores_list = list(algorithms_results.values())

    stats_data = []
    for alg_name, scores in algorithms_results.items():
        scores = np.array(scores)
        stats_data.append({
            'Algorithm': alg_name.upper(),
            'Mean': np.mean(scores),
            'Std': np.std(scores, ddof=1),
            'Median': np.median(scores),
            'Min': np.min(scores),
            'Max': np.max(scores),
            'N': len(scores)
        })

    df = pd.DataFrame(stats_data)

    df['Rank'] = df['Mean'].rank(ascending=False)

    df['Mean ± Std'] = df.apply(lambda row: f"{row['Mean']:.4f} ± {row['Std']:.4f}", axis=1)

    if len(scores_list[0]) >= 3 and len(algorithms) >= 3:
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*scores_list)

            df['Friedman_p'] = friedman_p
            df['Significant_Difference'] = 'Yes' if friedman_p < alpha else 'No'

            if friedman_p < alpha and POSTHOCS_AVAILABLE:
                data_matrix = np.array(scores_list).T
                df_posthoc = pd.DataFrame(data_matrix, columns=algorithms)
                posthoc_matrix = sp.posthoc_nemenyi_friedman(df_posthoc)

                for i, alg in enumerate(algorithms):
                    significant_diffs = []
                    for j, other_alg in enumerate(algorithms):
                        if i != j and posthoc_matrix.iloc[i, j] < alpha:
                            significant_diffs.append(other_alg.upper())

                    df.loc[i, 'Significantly_Different_From'] = ', '.join(significant_diffs) if significant_diffs else 'None'

        except Exception as e:
            print(f"Warning: Could not perform statistical tests: {e}")
            df['Friedman_p'] = np.nan
            df['Significant_Difference'] = 'Unknown'

    column_order = ['Algorithm', 'Mean ± Std', 'Median', 'Rank', 'Min', 'Max', 'N']
    if 'Significant_Difference' in df.columns:
        column_order.extend(['Significant_Difference', 'Significantly_Different_From'])

    df = df[column_order]
    df = df.sort_values('Rank')

    return df

def wilcoxon_signed_rank_test(scores1, scores2, alternative='two-sided'):

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    if len(scores1) != len(scores2):
        raise ValueError("Both score arrays must have the same length")

    statistic, p_value = stats.wilcoxon(scores1, scores2, alternative=alternative)

    z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
    effect_size = abs(z_score) / np.sqrt(len(scores1))

    if effect_size < 0.1:
        effect_interpretation = "negligible"
    elif effect_size < 0.3:
        effect_interpretation = "small"
    elif effect_size < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_interpretation': effect_interpretation,
        'significant': p_value < 0.05,
        'median_difference': np.median(scores1 - scores2)
    }
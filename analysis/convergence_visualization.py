import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

DATASETS = [
    ('iris', 'Iris'),
    ('titanic', 'Titanic'),
    ('beer', 'Beer'),
    ('polynomial', 'Polynomial'),
    ('sinusoidal', 'Sinusoidal'),
    ('gaussian_mixture', 'Gaussian Mix')
]

ALGORITHMS = ['sgd', 'scg', 'leapfrog']
ALGO_LABELS = ['SGD', 'SCG', 'LeapFrog']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

def load_convergence_data(dataset_name):
    results_dir = Path(__file__).parent.parent / 'results' / 'hyperparameter_tuning'
    pkl_file = results_dir / f'{dataset_name}_convergence_data.pkl'

    if not pkl_file.exists():
        warnings.warn(f"File not found: {pkl_file}")
        return None

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    return data

def plot_convergence_time():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (dataset_id, dataset_name) in enumerate(DATASETS):
        ax = axes[idx]
        data = load_convergence_data(dataset_id)

        if data is None:
            ax.text(0.5, 0.5, f'{dataset_name}\nData not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset_name, fontweight='bold')
            continue

        times = []
        errors = []
        labels = []

        for algo, label in zip(ALGORITHMS, ALGO_LABELS):
            if algo in data:
                conv_time = data[algo].get('convergence_time_mean')
                if conv_time is not None:
                    times.append(conv_time)
                    errors.append(0.01)
                    labels.append(label)
                else:
                    times.append(0)
                    errors.append(0)
                    labels.append(label)

        x = np.arange(len(labels))
        bars = ax.bar(x, times, color=COLORS[:len(labels)], alpha=0.8, edgecolor='black')

        for i, (bar, time) in enumerate(zip(bars, times)):
            if time > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.3f}s', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Convergence Time (s)', fontweight='bold')
        ax.set_title(dataset_name, fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/plots/convergence_time_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_convergence_epochs():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (dataset_id, dataset_name) in enumerate(DATASETS):
        ax = axes[idx]
        data = load_convergence_data(dataset_id)

        if data is None:
            ax.text(0.5, 0.5, f'{dataset_name}\nData not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset_name, fontweight='bold')
            continue

        epochs = []
        labels = []

        for algo, label in zip(ALGORITHMS, ALGO_LABELS):
            if algo in data:
                conv_epoch = data[algo].get('convergence_epoch_mean')
                if conv_epoch is not None:
                    epochs.append(conv_epoch)
                    labels.append(label)
                else:
                    epochs.append(0)
                    labels.append(label)

        x = np.arange(len(labels))
        bars = ax.bar(x, epochs, color=COLORS[:len(labels)], alpha=0.8, edgecolor='black')

        for i, (bar, epoch) in enumerate(zip(bars, epochs)):
            if epoch > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{epoch:.1f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Convergence Epoch', fontweight='bold')
        ax.set_title(dataset_name, fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/plots/convergence_epochs_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_efficiency_ratio():
    fig, ax = plt.subplots(figsize=(12, 6))

    all_data = []

    for dataset_id, dataset_name in DATASETS:
        data = load_convergence_data(dataset_id)
        if data is None:
            continue

        for algo, label, color in zip(ALGORITHMS, ALGO_LABELS, COLORS):
            if algo not in data:
                continue

            cv_scores = data[algo].get('cv_scores', [])
            if not cv_scores:
                continue

            mean_score = np.mean(cv_scores)

            conv_time = data[algo].get('convergence_time_mean')
            if conv_time is None or conv_time == 0:
                continue

            if dataset_id in ['polynomial', 'sinusoidal', 'gaussian_mixture']:
                efficiency = (1.0 / mean_score) / conv_time if mean_score > 0 else 0
            else:
                efficiency = mean_score / conv_time

            all_data.append({
                'Dataset': dataset_name,
                'Algorithm': label,
                'Efficiency': efficiency,
                'Color': color
            })

    df = pd.DataFrame(all_data)

    if df.empty:
        print("⚠️  No data available for efficiency plot")
        return

    x_pos = np.arange(len(DATASETS))
    width = 0.25

    for i, (algo, color) in enumerate(zip(ALGO_LABELS, COLORS)):
        algo_data = df[df['Algorithm'] == algo]

        values = []
        for dataset_id, dataset_name in DATASETS:
            row = algo_data[algo_data['Dataset'] == dataset_name]
            if len(row) > 0:
                values.append(row['Efficiency'].values[0])
            else:
                values.append(0)

        offset = width * (i - 1)
        ax.bar(x_pos + offset, values, width, label=algo, color=color, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Efficiency Ratio (Score / Time)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax.set_title('Algorithmic Efficiency Comparison\n(Higher is Better)', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name for _, name in DATASETS], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/plots/efficiency_ratio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_epoch_time_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))

    all_times = []

    for dataset_id, dataset_name in DATASETS:
        data = load_convergence_data(dataset_id)
        if data is None:
            continue

        for algo, label in zip(ALGORITHMS, ALGO_LABELS):
            if algo not in data:
                continue

            total_time = data[algo].get('total_time_mean')
            epoch_times = data[algo].get('epoch_times', [])

            if epoch_times:
                for t in epoch_times[:100]:
                    all_times.append({
                        'Algorithm': label,
                        'Epoch Time (ms)': t * 1000
                    })

    if not all_times:
        print("⚠️  No epoch time data available")
        return

    df = pd.DataFrame(all_times)

    sns.violinplot(data=df, x='Algorithm', y='Epoch Time (ms)', palette=COLORS, ax=ax, inner='box')

    ax.set_ylabel('Per-Epoch Time (ms)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
    ax.set_title('Computational Overhead per Iteration', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/plots/epoch_time_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("CONVERGENCE VISUALIZATION")
    print("="*80)
    print()

    Path('results/plots').mkdir(parents=True, exist_ok=True)

    print("Generating plots...\n")

    try:
        print("[1/4] Time-to-convergence comparison...")
        plot_convergence_time()
    except Exception as e:
        print(f"⚠️  Error: {e}")

    try:
        print("[2/4] Epochs-to-convergence comparison...")
        plot_convergence_epochs()
    except Exception as e:
        print(f"⚠️  Error: {e}")

    try:
        print("[3/4] Efficiency ratio analysis...")
        plot_efficiency_ratio()
    except Exception as e:
        print(f"⚠️  Error: {e}")

    try:
        print("[4/4] Per-epoch time distribution...")
        plot_epoch_time_distribution()
    except Exception as e:
        print(f"⚠️  Error: {e}")

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)

if __name__ == '__main__':
    main()
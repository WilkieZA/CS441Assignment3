import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold

from src.neural_net.torch_network import SingleHiddenLayerNet
from src.algorithms.torch_sgd import SGDOptimizer
from src.algorithms.torch_scg import SCGOptimizer
from src.algorithms.torch_leapfrog import LeapFrogOptimizer
from src.utils.convergence_metrics import ConvergenceTracker


def train_with_loss_tracking_single_fold(model, optimizer, criterion, X_train, y_train, X_val, y_val, max_epochs=2000, convergence_threshold=0.05):
    train_losses = []
    val_losses = []

    tracker = ConvergenceTracker(
        convergence_threshold=convergence_threshold,
        stability_window=5
    )

    import time
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 200

    for epoch in range(max_epochs):
        epoch_start = time.time()

        model.train()
        def closure():
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            return loss

        train_loss = optimizer.step(closure)

        if not torch.isfinite(train_loss):
            break

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        train_losses.append(train_loss.item() if torch.is_tensor(train_loss) else train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        tracker.add_epoch(val_loss, epoch_time)

        if tracker.is_converged():
            break

        if val_loss > 100 or not np.isfinite(val_loss):
            break

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tracker.mark_patience_converged()
                break

    convergence_epoch = tracker.get_convergence_epoch()
    return train_losses, val_losses, convergence_epoch


def train_with_loss_tracking_average(config, X, y, criterion, max_epochs, convergence_threshold, n_folds=10, is_classification=True):
    all_val_losses = []
    all_conv_epochs = []

    if is_classification:
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_iter = kfold.split(X, y)
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_iter = kfold.split(X, y)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std

        if not is_classification:
            y_mean, y_std = y_train.mean(), y_train.std()
            if y_std < 1e-8:
                y_std = 1.0
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        if is_classification:
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_val_t = torch.tensor(y_val, dtype=torch.long)
            input_dim = X.shape[1]
            output_dim = len(np.unique(y))
            model = SingleHiddenLayerNet(input_dim, config['hidden'], output_dim,
                                        hidden_activation='relu', output_activation='linear')
        else:
            y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            input_dim = X.shape[1]
            output_dim = 1
            model = SingleHiddenLayerNet(input_dim, config['hidden'], output_dim,
                                        hidden_activation='tanh', output_activation='linear')

        optimizer = config['class'](model.parameters(), **config['params'])

        train_losses, val_losses, conv_epoch = train_with_loss_tracking_single_fold(
            model, optimizer, criterion, X_train_t, y_train_t, X_val_t, y_val_t,
            max_epochs=max_epochs, convergence_threshold=convergence_threshold
        )

        all_val_losses.append(val_losses)
        all_conv_epochs.append(conv_epoch)

    max_len = max(len(losses) for losses in all_val_losses)
    padded_losses = []
    for losses in all_val_losses:
        padded = losses + [np.nan] * (max_len - len(losses))
        padded_losses.append(padded)

    avg_losses = np.nanmean(padded_losses, axis=0)

    valid_conv = [e for e in all_conv_epochs if e is not None]
    avg_conv_epoch = int(np.mean(valid_conv)) if valid_conv else None

    return avg_losses, avg_conv_epoch


def plot_beer_classification_convergence():
    print("\n" + "="*80)
    print("BEER CLASSIFICATION - Loss Curve Visualization")
    print("="*80)

    X = np.load('data/processed/beer_X_train.npy')
    y = np.load('data/processed/beer_y_train.npy')

    configs = {
        'SGD': {
            'class': SGDOptimizer,
            'params': {'lr': 0.1, 'momentum': 0.0},
            'hidden': 25,
            'color': '#1f77b4',
            'label': 'SGD (lr=0.1, μ=0.0, h=25)'
        },
        'SCG': {
            'class': SCGOptimizer,
            'params': {'sigma': 1e-6, 'lambd': 1e-3},
            'hidden': 22,
            'color': '#ff7f0e',
            'label': 'SCG (σ=1e-6, λ=1e-3, h=22)'
        },
        'LeapFrog': {
            'class': LeapFrogOptimizer,
            'params': {'dt': 1e-4, 'delta_max': 1.0, 'xi': 0.2, 'm': 5},
            'hidden': 30,
            'color': '#2ca02c',
            'label': 'LeapFrog (Δt=1e-4, δ=1.0, h=30)'
        }
    }

    criterion = nn.CrossEntropyLoss()
    convergence_threshold = 0.15
    max_epochs = 1000

    plt.figure(figsize=(12, 7))

    for algo_name, config in configs.items():
        print(f"\nTraining {algo_name} (averaging across 10 folds)...")

        val_losses, conv_epoch = train_with_loss_tracking_average(
            config, X, y, criterion, max_epochs, convergence_threshold,
            n_folds=10, is_classification=True
        )

        valid_losses = val_losses[~np.isnan(val_losses)]

        if conv_epoch is not None and conv_epoch <= len(valid_losses):
            plot_losses = valid_losses[:conv_epoch]
            plt.plot(plot_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85)

            plt.scatter(conv_epoch-1, valid_losses[conv_epoch-1], color=config['color'],
                       s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        else:
            plt.plot(valid_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85, linestyle='--')

        print(f"Averaged final validation loss: {valid_losses[-1]:.4f}")
        print(f"Averaged epochs trained: {len(valid_losses)}")
        if conv_epoch:
            print(f"Averaged convergence at epoch: {conv_epoch}")

    plt.xlabel('Epoch', fontweight='bold', fontsize=12)
    plt.ylabel('Validation Loss (Cross-Entropy)', fontweight='bold', fontsize=12)
    plt.title('Beer Classification: Training Convergence Comparison\n(17-class multi-class problem)',
              fontweight='bold', fontsize=14)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = Path('results/plots/beer_loss_curves.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_gaussian_mixture_convergence():
    print("\n" + "="*80)
    print("GAUSSIAN MIXTURE REGRESSION - Loss Curve Visualization")
    print("="*80)

    X = np.load('data/synthetic/gaussian_mixture_X.npy')
    y = np.load('data/synthetic/gaussian_mixture_y.npy')

    configs = {
        'SGD': {
            'class': SGDOptimizer,
            'params': {'lr': 0.1, 'momentum': 0.9},
            'hidden': 29,
            'color': '#1f77b4',
            'label': 'SGD (lr=0.1, μ=0.9, h=29)'
        },
        'SCG': {
            'class': SCGOptimizer,
            'params': {'sigma': 1e-5, 'lambd': 1e-3},
            'hidden': 24,
            'color': '#ff7f0e',
            'label': 'SCG (σ=1e-5, λ=1e-3, h=24)'
        },
        'LeapFrog': {
            'class': LeapFrogOptimizer,
            'params': {'dt': 1e-3, 'delta_max': 0.1, 'xi': 0.2, 'm': 3},
            'hidden': 34,
            'color': '#2ca02c',
            'label': 'LeapFrog (Δt=1e-3, δ=0.1, h=34)'
        }
    }

    criterion = nn.MSELoss()
    convergence_threshold = 0.15
    max_epochs = 2000

    plt.figure(figsize=(12, 7))

    for algo_name, config in configs.items():
        print(f"\nTraining {algo_name} (averaging across 10 folds)...")

        val_losses, conv_epoch = train_with_loss_tracking_average(
            config, X, y, criterion, max_epochs, convergence_threshold,
            n_folds=10, is_classification=False
        )

        valid_losses = val_losses[~np.isnan(val_losses)]

        if conv_epoch is not None and conv_epoch <= len(valid_losses):
            plot_losses = valid_losses[:conv_epoch]
            plt.plot(plot_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85)

            plt.scatter(conv_epoch-1, valid_losses[conv_epoch-1], color=config['color'],
                       s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        else:
            plt.plot(valid_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85, linestyle='--')

        print(f"  Averaged final validation loss: {valid_losses[-1]:.4f}")
        print(f"  Averaged epochs trained: {len(valid_losses)}")
        if conv_epoch:
            print(f"  Averaged convergence at epoch: {conv_epoch}")

    plt.xlabel('Epoch', fontweight='bold', fontsize=12)
    plt.ylabel('Validation Loss (MSE)', fontweight='bold', fontsize=12)
    plt.title('Gaussian Mixture Regression: Training Convergence Comparison\n(3D multi-modal surface)',
              fontweight='bold', fontsize=14)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = Path('results/plots/gaussian_mixture_loss_curves.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_iris_classification_convergence():
    print("\n" + "="*80)
    print("IRIS CLASSIFICATION - Loss Curve Visualization")
    print("="*80)

    X = np.load('data/processed/iris_X_train.npy')
    y = np.load('data/processed/iris_y_train.npy')

    configs = {
        'SGD': {
            'class': SGDOptimizer,
            'params': {'lr': 0.1, 'momentum': 0.99},
            'hidden': 30,
            'color': '#1f77b4',
            'label': 'SGD (lr=0.1, μ=0.99, h=30)'
        },
        'SCG': {
            'class': SCGOptimizer,
            'params': {'sigma': 1e-6, 'lambd': 1e-2},
            'hidden': 8,
            'color': '#ff7f0e',
            'label': 'SCG (σ=1e-6, λ=1e-2, h=8)'
        },
        'LeapFrog': {
            'class': LeapFrogOptimizer,
            'params': {'dt': 1e-4, 'delta_max': 1.0, 'xi': 0.1, 'm': 5},
            'hidden': 12,
            'color': '#2ca02c',
            'label': 'LeapFrog (Δt=1e-4, δ=1.0, h=12)'
        }
    }

    criterion = nn.CrossEntropyLoss()
    convergence_threshold = 0.05
    max_epochs = 500

    plt.figure(figsize=(12, 7))

    for algo_name, config in configs.items():
        print(f"\nTraining {algo_name} (averaging across 10 folds)...")

        val_losses, conv_epoch = train_with_loss_tracking_average(
            config, X, y, criterion, max_epochs, convergence_threshold,
            n_folds=10, is_classification=True
        )

        valid_losses = val_losses[~np.isnan(val_losses)]

        if conv_epoch is not None and conv_epoch <= len(valid_losses):
            plot_losses = valid_losses[:conv_epoch]
            plt.plot(plot_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85)

            plt.scatter(conv_epoch-1, valid_losses[conv_epoch-1], color=config['color'],
                       s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        else:
            plt.plot(valid_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85, linestyle='--')

        print(f"Averaged final validation loss: {valid_losses[-1]:.4f}")
        print(f"Averaged epochs trained: {len(valid_losses)}")
        if conv_epoch:
            print(f"Averaged convergence at epoch: {conv_epoch}")

    plt.xlabel('Epoch', fontweight='bold', fontsize=12)
    plt.ylabel('Validation Loss (Cross-Entropy)', fontweight='bold', fontsize=12)
    plt.title('Iris Classification: Training Convergence Comparison\n(3-class simple problem)',
              fontweight='bold', fontsize=14)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = Path('results/plots/iris_loss_curves.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_polynomial_convergence():
    print("\n" + "="*80)
    print("POLYNOMIAL REGRESSION - Loss Curve Visualization")
    print("="*80)

    X = np.load('data/synthetic/polynomial_X.npy')
    y = np.load('data/synthetic/polynomial_y.npy')

    configs = {
        'SGD': {
            'class': SGDOptimizer,
            'params': {'lr': 0.05, 'momentum': 0.99},
            'hidden': 23,
            'color': '#1f77b4',
            'label': 'SGD (lr=0.05, μ=0.99, h=23)'
        },
        'SCG': {
            'class': SCGOptimizer,
            'params': {'sigma': 1e-6, 'lambd': 1e-2},
            'hidden': 16,
            'color': '#ff7f0e',
            'label': 'SCG (σ=1e-6, λ=1e-2, h=16)'
        },
        'LeapFrog': {
            'class': LeapFrogOptimizer,
            'params': {'dt': 1e-3, 'delta_max': 1.0, 'xi': 0.1, 'm': 3},
            'hidden': 30,
            'color': '#2ca02c',
            'label': 'LeapFrog (Δt=1e-3, δ=1.0, h=30)'
        }
    }

    criterion = nn.MSELoss()
    convergence_threshold = 0.05
    max_epochs = 500

    plt.figure(figsize=(12, 7))

    for algo_name, config in configs.items():
        print(f"\nTraining {algo_name} (averaging across 10 folds)...")

        val_losses, conv_epoch = train_with_loss_tracking_average(
            config, X, y, criterion, max_epochs, convergence_threshold,
            n_folds=10, is_classification=False
        )

        valid_losses = val_losses[~np.isnan(val_losses)]

        if conv_epoch is not None and conv_epoch <= len(valid_losses):
            plot_losses = valid_losses[:conv_epoch]
            plt.plot(plot_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85)

            plt.scatter(conv_epoch-1, valid_losses[conv_epoch-1], color=config['color'],
                       s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        else:
            plt.plot(valid_losses, color=config['color'], linewidth=2,
                    label=config['label'], alpha=0.85, linestyle='--')

        print(f"Averaged final validation loss: {valid_losses[-1]:.4f}")
        print(f"Averaged epochs trained: {len(valid_losses)}")
        if conv_epoch:
            print(f"Averaged convergence at epoch: {conv_epoch}")

    plt.xlabel('Epoch', fontweight='bold', fontsize=12)
    plt.ylabel('Validation Loss (MSE)', fontweight='bold', fontsize=12)
    plt.title('Polynomial Regression: Training Convergence Comparison\n(1D cubic function)',
              fontweight='bold', fontsize=14)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = Path('results/plots/polynomial_loss_curves.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("LOSS CURVE VISUALIZATION - SIMPLE & COMPLEX TASKS")
    print("="*80)
    print("\nGenerating convergence plots for optimal configurations...")

    try:
        plot_iris_classification_convergence()
    except Exception as e:
        print(f"\n⚠️  Error plotting Iris classification: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_polynomial_convergence()
    except Exception as e:
        print(f"\n⚠️  Error plotting Polynomial regression: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_beer_classification_convergence()
    except Exception as e:
        print(f"\n⚠️  Error plotting Beer classification: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_gaussian_mixture_convergence()
    except Exception as e:
        print(f"\n⚠️  Error plotting Gaussian mixture: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Loss curve visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()

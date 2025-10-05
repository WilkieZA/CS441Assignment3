import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedKFold

from src.neural_net.torch_network import SingleHiddenLayerNet
from src.algorithms.torch_sgd import SGDOptimizer
from src.algorithms.torch_scg import SCGOptimizer
from src.algorithms.torch_leapfrog import LeapFrogOptimizer
from src.utils.convergence_metrics import ConvergenceTracker


@dataclass
class JointResult:
    algorithm: str
    params: Dict[str, Any]
    hidden_dim: int
    accuracy_mean: float
    accuracy_std: float
    cv_scores: List[float]
    convergence_epoch_mean: float
    convergence_time_mean: float
    total_time_mean: float
    epoch_times: List[List[float]]
    fold_val_histories: List[List[float]] = None
    fold_best_epochs: List[int] = None
    fold_best_losses: List[float] = None


def load_titanic_dataset():
    print("Loading Titanic Survival Classification dataset...")
    X = np.load('data/processed/titanic_X_train.npy')
    y = np.load('data/processed/titanic_y_train.npy')
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))} ({np.unique(y)})")
    return X, y


def evaluate_joint_config(optimizer_class, algorithm_name: str, params: Dict,
                        hidden_dim: int, X: np.ndarray, y: np.ndarray,
                        n_folds: int = 10, max_epochs: int = 10000, seed: int = 42,
                        convergence_threshold: float = 0.20, use_early_stopping: bool = True):

    n_classes = len(np.unique(y))
    input_dim = X.shape[1]

    skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores = []
    trackers = []
    all_epoch_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        model = SingleHiddenLayerNet(
            input_dim, hidden_dim, n_classes,
            hidden_activation='relu',
            output_activation='linear'  # Use logits for CrossEntropyLoss
        )

        optimizer = optimizer_class(model.parameters(), **params)

        criterion = nn.CrossEntropyLoss()

        tracker = ConvergenceTracker(
            convergence_threshold=convergence_threshold,
            stability_window=5
        )
        epoch_times_history = []
        best_val_epoch = 0
        val_loss_history = []  # Track all validation losses

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 100  # For SGD early stopping

        try:
            for epoch in range(max_epochs):
                epoch_start = time.time()

                model.train()
                def closure():
                    optimizer.zero_grad()
                    outputs = model(X_train_t)
                    loss = criterion(outputs, y_train_t)
                    return loss

                train_loss = optimizer.step(closure)

                if not torch.isfinite(train_loss):
                    raise ValueError(f"Non-finite loss at epoch {epoch}")

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                epoch_time = time.time() - epoch_start
                epoch_times_history.append(epoch_time)
                tracker.add_epoch(val_loss, epoch_time)
                val_loss_history.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if tracker.is_converged():
                    break

                if use_early_stopping and patience_counter >= patience:
                    tracker.mark_patience_converged()
                    break

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == y_val_t).float().mean().item()

            cv_scores.append(accuracy)
            trackers.append(tracker)
            all_epoch_times.append(epoch_times_history)

            if fold_idx == 0:  # Initialize on first fold
                fold_val_histories = []
                fold_best_epochs = []
                fold_best_losses = []
            fold_val_histories.append(val_loss_history)
            fold_best_epochs.append(best_val_epoch)
            fold_best_losses.append(best_val_loss)

        except Exception as e:
            print(f"Failed: {algorithm_name} h={hidden_dim} {params} - {str(e)}")
            cv_scores.append(0.0)
            dummy_tracker = ConvergenceTracker(convergence_threshold=convergence_threshold)
            trackers.append(dummy_tracker)
            all_epoch_times.append([])

    convergence_epochs = [tracker.get_convergence_epoch() for tracker in trackers]
    convergence_times = [tracker.get_convergence_time() for tracker in trackers]
    total_times = [sum(epoch_times) if epoch_times else 0.0 for epoch_times in all_epoch_times]

    valid_conv_epochs = [e for e in convergence_epochs if e is not None]
    valid_conv_times = [t for t in convergence_times if t is not None]

    convergence_epoch_mean = np.mean(valid_conv_epochs) if valid_conv_epochs else None
    convergence_time_mean = np.mean(valid_conv_times) if valid_conv_times else None
    total_time_mean = np.mean(total_times)

    return {
        'accuracy_mean': np.mean(cv_scores),
        'accuracy_std': np.std(cv_scores),
        'cv_scores': cv_scores,
        'convergence_epoch_mean': convergence_epoch_mean,
        'convergence_time_mean': convergence_time_mean,
        'total_time_mean': total_time_mean,
        'epoch_times': all_epoch_times,
        'fold_val_histories': fold_val_histories if 'fold_val_histories' in locals() else [],
        'fold_best_epochs': fold_best_epochs if 'fold_best_epochs' in locals() else [],
        'fold_best_losses': fold_best_losses if 'fold_best_losses' in locals() else []
    }


def phase1_coarse_search(X: np.ndarray, y: np.ndarray):
    print("\n" + "="*60)
    print("PHASE 1: COARSE JOINT GRID SEARCH (PyTorch)")
    print("="*60)

    architectures = [10, 15, 20, 25, 30]
    algorithms = {
        'sgd': (SGDOptimizer, {'lr': [0.001, 0.01, 0.05, 0.1], 'momentum': [0.0, 0.5, 0.9, 0.99]}, True),  # use early stopping
        'scg': (SCGOptimizer, {'sigma': [1e-6, 1e-5], 'lambd': [1e-3, 1e-2]}, False),  # no early stopping
        'leapfrog': (LeapFrogOptimizer, {'dt': [1e-4, 1e-3], 'delta_max': [0.1, 1.0],
                                         'xi': [0.1, 0.2], 'm': [3, 5]}, False)  # no early stopping
    }

    results = []
    total_configs = sum(len(list(itertools.product(*param_grid.values()))) * len(architectures)
                        for _, (_, param_grid, _) in algorithms.items())

    print(f"Testing {total_configs} total configurations:")
    for alg_name, (_, param_grid, _) in algorithms.items():
        n_params = len(list(itertools.product(*param_grid.values())))
        print(f"  {alg_name.upper()}: {n_params} params × {len(architectures)} arch = {n_params * len(architectures)} configs")

    config_idx = 0
    start_time = time.time()

    for alg_name, (alg_class, param_grid, use_early_stopping) in algorithms.items():
        param_combinations = [dict(zip(param_grid.keys(), combo))
                            for combo in itertools.product(*param_grid.values())]

        for params in param_combinations:
            for h in architectures:
                config_idx += 1
                print(f"[{config_idx:3d}/{total_configs}] {alg_name.upper()} h={h:2d} {params}")

                result_dict = evaluate_joint_config(
                    alg_class, alg_name, params, h, X, y, max_epochs=2000, use_early_stopping=use_early_stopping
                )

                result = JointResult(
                    algorithm=alg_name, params=params, hidden_dim=h,
                    accuracy_mean=result_dict['accuracy_mean'],
                    accuracy_std=result_dict['accuracy_std'],
                    cv_scores=result_dict['cv_scores'],
                    convergence_epoch_mean=result_dict['convergence_epoch_mean'],
                    convergence_time_mean=result_dict['convergence_time_mean'],
                    total_time_mean=result_dict['total_time_mean'],
                    epoch_times=result_dict['epoch_times'],
                    fold_val_histories=result_dict.get('fold_val_histories', []),
                    fold_best_epochs=result_dict.get('fold_best_epochs', []),
                    fold_best_losses=result_dict.get('fold_best_losses', [])
                )
                results.append(result)

                if result.accuracy_mean > 0.0:
                    conv_status = "✓" if result.convergence_epoch_mean is not None else "×"
                    conv_info = ""
                    if result.convergence_epoch_mean is not None:
                        conv_info = f" | Conv: {result.convergence_epoch_mean:.0f}ep, {result.convergence_time_mean:.2f}s"
                    print(f"    {conv_status} Accuracy: {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}{conv_info}")
                else:
                    print(f"    × FAILED")

    elapsed = time.time() - start_time
    print(f"\nPhase 1 completed in {elapsed:.1f} seconds")

    import pickle
    results_dir = Path("results/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "titanic_phase1_all.pkl", "wb") as f:
        pickle.dump([{'algorithm': r.algorithm, 'params': r.params, 'hidden_dim': r.hidden_dim,
                      'accuracy_mean': r.accuracy_mean, 'accuracy_std': r.accuracy_std}
                     for r in results], f)
    print(f"Saved all Phase 1 results for sensitivity analysis")

    valid_results = [r for r in results if r.accuracy_mean > 0.0]
    if not valid_results:
        raise ValueError("All configurations failed!")

    best = max(valid_results, key=lambda r: r.accuracy_mean)
    print(f"\nBest Phase 1 Result:")
    print(f"  Algorithm: {best.algorithm.upper()}")
    print(f"  Parameters: {best.params}")
    print(f"  Architecture: {best.hidden_dim} hidden units")
    print(f"  Accuracy: {best.accuracy_mean:.4f} ± {best.accuracy_std:.4f}")

    return best, valid_results


def phase2_fine_search_per_algorithm(phase1_results: List[JointResult], X: np.ndarray, y: np.ndarray):
    print("\n" + "="*60)
    print("PHASE 2: FINE GRID REFINEMENT (PER ALGORITHM)")
    print("="*60)

    algorithms = ['sgd', 'scg', 'leapfrog']
    alg_classes = {'sgd': SGDOptimizer, 'scg': SCGOptimizer, 'leapfrog': LeapFrogOptimizer}

    best_per_algorithm = {}
    for alg in algorithms:
        alg_results = [r for r in phase1_results if r.algorithm == alg and r.accuracy_mean > 0.0]
        if alg_results:
            best_per_algorithm[alg] = max(alg_results, key=lambda r: (r.accuracy_mean, -r.accuracy_std, -r.hidden_dim))

    print(f"Best Phase 1 results per algorithm:")
    for alg, best in best_per_algorithm.items():
        print(f"  {alg.upper()}: h={best.hidden_dim}, Accuracy={best.accuracy_mean:.4f}±{best.accuracy_std:.4f}")

    final_results = {}

    early_stopping_map = {'sgd': True, 'scg': False, 'leapfrog': False}

    for alg_name, best_phase1 in best_per_algorithm.items():
        print(f"\n--- Fine search for {alg_name.upper()} ---")

        best_h = best_phase1.hidden_dim
        fine_architectures = [max(5, best_h-4), max(5, best_h-2), best_h, best_h+2, best_h+4]
        fine_architectures = list(set(fine_architectures))
        fine_architectures.sort()

        print(f"Fine search architectures: {fine_architectures}")
        alg_class = alg_classes[alg_name]
        best_params = best_phase1.params

        phase2_results = []
        for h in fine_architectures:
            print(f"  Testing h={h:2d} with params {best_params}")

            result_dict = evaluate_joint_config(
                alg_class, alg_name, best_params, h, X, y, use_early_stopping=early_stopping_map[alg_name]
            )

            result = JointResult(
                algorithm=alg_name, params=best_params, hidden_dim=h,
                accuracy_mean=result_dict['accuracy_mean'],
                accuracy_std=result_dict['accuracy_std'],
                cv_scores=result_dict['cv_scores'],
                convergence_epoch_mean=result_dict['convergence_epoch_mean'],
                convergence_time_mean=result_dict['convergence_time_mean'],
                total_time_mean=result_dict['total_time_mean'],
                epoch_times=result_dict['epoch_times'],
                fold_val_histories=result_dict.get('fold_val_histories', []),
                fold_best_epochs=result_dict.get('fold_best_epochs', []),
                fold_best_losses=result_dict.get('fold_best_losses', [])
            )
            phase2_results.append(result)

            if result.accuracy_mean > 0.0:
                conv_status = "✓" if result.convergence_epoch_mean is not None else "×"
                conv_info = ""
                if result.convergence_epoch_mean is not None:
                    conv_info = f" | Conv: {result.convergence_epoch_mean:.0f}ep, {result.convergence_time_mean:.2f}s"
                print(f"    {conv_status} Accuracy: {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}{conv_info}")
            else:
                print(f"    × FAILED")

        valid_results = [r for r in phase2_results if r.accuracy_mean > 0.0]
        if valid_results:
            best_alg_final = max(valid_results, key=lambda r: (r.accuracy_mean, -r.accuracy_std, -r.hidden_dim))
            final_results[alg_name] = best_alg_final
            print(f"  Best {alg_name.upper()}: h={best_alg_final.hidden_dim}, Accuracy={best_alg_final.accuracy_mean:.4f}±{best_alg_final.accuracy_std:.4f}")
        else:
            final_results[alg_name] = best_phase1
            print(f"  Phase 2 failed for {alg_name.upper()}, keeping Phase 1 best")

    return final_results


def main():
    print("JOINT HYPERPARAMETER AND ARCHITECTURE OPTIMIZATION")
    print("Titanic Survival Classification (PyTorch)")
    print("="*60)

    X, y = load_titanic_dataset()

    best_phase1, phase1_results = phase1_coarse_search(X, y)

    algorithm_results = phase2_fine_search_per_algorithm(phase1_results, X, y)

    print("\n" + "="*60)
    print("ALGORITHM COMPARISON - JOINT OPTIMIZATION RESULTS")
    print("="*60)

    sorted_algorithms = sorted(algorithm_results.items(), key=lambda x: x[1].accuracy_mean, reverse=True)

    print(f"{'Rank':<4} {'Algorithm':<12} {'Hidden':<7} {'Accuracy':<15} {'Conv_Epoch':<12} {'Conv_Time(s)':<14} {'Total_Time(s)':<14}")
    print("-" * 100)

    for rank, (alg_name, result) in enumerate(sorted_algorithms, 1):
        conv_epoch_str = f"{result.convergence_epoch_mean:.1f}" if result.convergence_epoch_mean is not None else "N/A"
        conv_time_str = f"{result.convergence_time_mean:.3f}" if result.convergence_time_mean is not None else "N/A"
        total_time_str = f"{result.total_time_mean:.2f}"
        print(f"{rank:<4} {alg_name.upper():<12} {result.hidden_dim:<7} "
              f"{result.accuracy_mean:.4f}±{result.accuracy_std:.4f} {conv_epoch_str:<12} {conv_time_str:<14} {total_time_str:<14}")

    best_alg, best_result = sorted_algorithms[0]
    print(f"\nOverall Best Configuration:")
    print(f"  Algorithm: {best_alg.upper()}")
    print(f"  Parameters: {best_result.params}")
    print(f"  Hidden Units: {best_result.hidden_dim}")
    print(f"  Final Accuracy: {best_result.accuracy_mean:.4f} ± {best_result.accuracy_std:.4f}")

    results_dir = Path("results/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "joint_titanic_optimization.txt", "w") as f:
        f.write("JOINT HYPERPARAMETER AND ARCHITECTURE OPTIMIZATION - TITANIC\n")
        f.write("="*60 + "\n\n")
        f.write("Methodology: Two-phase per-algorithm optimization\n")
        f.write("Phase 1: Coarse grid over parameters and architecture\n")
        f.write("Phase 2: Fine architecture search per algorithm with optimal parameters\n\n")

        f.write(f"Overall Best Configuration:\n")
        f.write(f"  Algorithm: {best_alg.upper()}\n")
        f.write(f"  Parameters: {best_result.params}\n")
        f.write(f"  Hidden Units: {best_result.hidden_dim}\n")
        f.write(f"  Accuracy: {best_result.accuracy_mean:.4f} ± {best_result.accuracy_std:.4f}\n\n")

        f.write("Per-Algorithm Final Results:\n")
        for rank, (alg_name, result) in enumerate(sorted_algorithms, 1):
            f.write(f"  {rank}. {alg_name.upper()}: h={result.hidden_dim}, ")
            f.write(f"Accuracy={result.accuracy_mean:.4f}±{result.accuracy_std:.4f}\n")
            f.write(f"      Parameters: {result.params}\n")
            if result.convergence_epoch_mean is not None:
                f.write(f"      Convergence Epoch: {result.convergence_epoch_mean:.1f}\n")
            if result.convergence_time_mean is not None:
                f.write(f"      Convergence Time: {result.convergence_time_mean:.3f}s\n")
            f.write(f"      Total Training Time: {result.total_time_mean:.2f}s\n")

        f.write(f"\nPhase 1 Summary ({len(phase1_results)} configurations tested):\n")
        valid_phase1 = [r for r in phase1_results if r.accuracy_mean > 0.0]
        sorted_phase1 = sorted(valid_phase1, key=lambda r: r.accuracy_mean, reverse=True)

        for i, result in enumerate(sorted_phase1[:10]):
            f.write(f"  {i+1:2d}. {result.algorithm.upper()} h={result.hidden_dim} ")
            f.write(f"Accuracy={result.accuracy_mean:.4f}±{result.accuracy_std:.4f}\n")

    import pickle
    convergence_data = {
        alg_name: {
            'convergence_epoch_mean': result.convergence_epoch_mean,
            'convergence_time_mean': result.convergence_time_mean,
            'total_time_mean': result.total_time_mean,
            'epoch_times': result.epoch_times,
            'cv_scores': result.cv_scores,
            'fold_val_histories': result.fold_val_histories if result.fold_val_histories else [],
            'fold_best_epochs': result.fold_best_epochs if result.fold_best_epochs else [],
            'fold_best_losses': result.fold_best_losses if result.fold_best_losses else []
        } for alg_name, result in algorithm_results.items()
    }
    with open(results_dir / "titanic_convergence_data.pkl", "wb") as f:
        pickle.dump(convergence_data, f)

    print(f"\nResults saved to: {results_dir / 'joint_titanic_optimization.txt'}")
    print(f"Convergence data saved to: {results_dir / 'titanic_convergence_data.pkl'}")
    print("Joint optimization complete! 🎯")


if __name__ == "__main__":
    main()
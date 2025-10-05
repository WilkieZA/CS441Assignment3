import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ConvergenceTracker:

    def __init__(self, convergence_threshold: float = 0.05, stability_window: int = 5):
        self.convergence_threshold = convergence_threshold
        self.stability_window = stability_window

        self.val_losses = []
        self.epoch_times = []
        self.cumulative_times = []
        self.convergence_epoch = None
        self.convergence_time = None
        self._converged = False

        self.best_epoch = None
        self.best_loss = float('inf')
        self.best_time = None

    def add_epoch(self, val_loss: float, epoch_time: float) -> None:
        self.val_losses.append(val_loss)
        self.epoch_times.append(epoch_time)

        total_time = sum(self.epoch_times)
        self.cumulative_times.append(total_time)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = len(self.val_losses)  # 1-indexed
            self.best_time = total_time

        if not self._converged and len(self.val_losses) >= self.stability_window:
            self._check_convergence()

    def _check_convergence(self) -> None:
        if len(self.val_losses) < self.stability_window:
            return

        recent_losses = self.val_losses[-self.stability_window:]

        if all(loss <= self.convergence_threshold for loss in recent_losses):
            self.convergence_epoch = len(self.val_losses) - self.stability_window + 1
            self.convergence_time = self.cumulative_times[self.convergence_epoch - 1]
            self._converged = True

    def mark_patience_converged(self) -> None:
        if not self._converged and self.best_epoch is not None:
            self.convergence_epoch = self.best_epoch
            self.convergence_time = self.best_time
            self._converged = True

    def is_converged(self) -> bool:
        return self._converged

    def get_convergence_epoch(self) -> Optional[int]:
        return self.convergence_epoch

    def get_convergence_time(self) -> Optional[float]:
        return self.convergence_time

    def get_stats(self) -> Dict:
        total_epochs = len(self.val_losses)
        total_time = sum(self.epoch_times)

        stats = {
            'converged': self._converged,
            'convergence_epoch': self.convergence_epoch,
            'convergence_time': self.convergence_time,
            'convergence_threshold': self.convergence_threshold,
            'stability_window': self.stability_window,
            'total_epochs': total_epochs,
            'total_time': total_time,
            'final_loss': self.val_losses[-1] if self.val_losses else None,
            'min_loss': min(self.val_losses) if self.val_losses else None,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else None,
            'std_epoch_time': np.std(self.epoch_times) if self.epoch_times else None,
            'val_losses': self.val_losses.copy(),
            'epoch_times': self.epoch_times.copy(),
            'cumulative_times': self.cumulative_times.copy(),
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss if self.best_loss != float('inf') else None,
            'best_time': self.best_time
        }

        if self._converged and total_epochs > 0:
            stats['convergence_efficiency'] = self.convergence_epoch / total_epochs
            stats['time_efficiency'] = self.convergence_time / total_time
        else:
            stats['convergence_efficiency'] = None
            stats['time_efficiency'] = None

        if self.best_epoch is not None and total_epochs > 0:
            stats['best_epoch_efficiency'] = self.best_epoch / total_epochs
            stats['best_time_efficiency'] = self.best_time / total_time if total_time > 0 else None
        else:
            stats['best_epoch_efficiency'] = None
            stats['best_time_efficiency'] = None

        return stats


def compute_time_to_convergence(
    val_losses: List[float],
    epoch_times: List[float],
    threshold: float = 0.05,
    window: int = 5
) -> Tuple[Optional[int], Optional[float]]:

    if len(val_losses) != len(epoch_times):
        raise ValueError("val_losses and epoch_times must have the same length")

    if len(val_losses) < window:
        return None, None

    cumulative_times = np.cumsum(epoch_times)

    for i in range(len(val_losses) - window + 1):
        window_losses = val_losses[i:i + window]

        if all(loss <= threshold for loss in window_losses):
            convergence_epoch = i + 1  # 1-indexed
            convergence_time = cumulative_times[i + window - 1]  # Time at end of window
            return convergence_epoch, convergence_time

    return None, None


def compare_convergence_efficiency(
    results_dict: Dict[str, Dict[str, List[float]]],
    threshold: float = 0.05,
    window: int = 5
) -> pd.DataFrame:

    comparison_data = []

    for algorithm, data in results_dict.items():
        val_losses = data['val_losses']
        epoch_times = data['epoch_times']

        if len(val_losses) != len(epoch_times):
            warnings.warn(f"Mismatched data lengths for {algorithm}: "
                         f"{len(val_losses)} losses vs {len(epoch_times)} times")
            continue

        if len(val_losses) == 0:
            warnings.warn(f"No data for algorithm {algorithm}")
            continue

        conv_epoch, conv_time = compute_time_to_convergence(
            val_losses, epoch_times, threshold, window
        )

        total_epochs = len(val_losses)
        total_time = sum(epoch_times)
        avg_epoch_time = np.mean(epoch_times)
        final_loss = val_losses[-1]
        min_loss = min(val_losses)

        if conv_time is not None and total_time > 0:
            efficiency_ratio = conv_time / total_time
        else:
            efficiency_ratio = None

        row_data = {
            'Algorithm': algorithm.upper(),
            'Converged': conv_epoch is not None,
            'Convergence_Epoch': conv_epoch,
            'Convergence_Time': conv_time,
            'Avg_Epoch_Time': avg_epoch_time,
            'Total_Epochs': total_epochs,
            'Total_Time': total_time,
            'Final_Loss': final_loss,
            'Min_Loss': min_loss,
            'Efficiency_Ratio': efficiency_ratio
        }

        comparison_data.append(row_data)

    df = pd.DataFrame(comparison_data)

    if len(df) > 0:
        df['sort_key'] = df.apply(
            lambda row: (0, row['Convergence_Time']) if row['Converged']
            else (1, row['Total_Time']), axis=1
        )
        df = df.sort_values('sort_key').drop('sort_key', axis=1)

    return df


def analyze_convergence_patterns(
    val_losses: List[float],
    epoch_times: List[float],
    algorithm_name: str = "Algorithm"
) -> Dict:

    val_losses = np.array(val_losses)
    epoch_times = np.array(epoch_times)

    if len(val_losses) < 5:
        return {
            'algorithm': algorithm_name,
            'convergence_rate': None,
            'stability_score': None,
            'plateau_detected': False,
            'improvement_rate': None,
            'early_stopping_recommended': False
        }

    analysis = {
        'algorithm': algorithm_name,
        'total_epochs': len(val_losses)
    }

    early_phase = max(5, len(val_losses) // 4)
    early_losses = val_losses[:early_phase]
    if len(early_losses) > 1:
        epochs = np.arange(len(early_losses))
        slope, _ = np.polyfit(epochs, early_losses, 1)
        analysis['convergence_rate'] = abs(slope)  # Loss reduction per epoch
    else:
        analysis['convergence_rate'] = None

    final_phase = max(5, len(val_losses) // 4)
    final_losses = val_losses[-final_phase:]
    if len(final_losses) > 1 and np.mean(final_losses) > 0:
        cv = np.std(final_losses) / np.mean(final_losses)
        analysis['stability_score'] = cv
    else:
        analysis['stability_score'] = None

    recent_window = min(20, len(val_losses) // 2)
    recent_losses = val_losses[-recent_window:]

    if len(recent_losses) > 5:
        epochs = np.arange(len(recent_losses))
        slope, _ = np.polyfit(epochs, recent_losses, 1)

        relative_slope = abs(slope) / (np.mean(recent_losses) + 1e-8)
        analysis['plateau_detected'] = relative_slope < 0.001
        analysis['improvement_rate'] = slope  # Negative = improvement
    else:
        analysis['plateau_detected'] = False
        analysis['improvement_rate'] = None

    if analysis['plateau_detected'] and analysis['stability_score'] is not None:
        analysis['early_stopping_recommended'] = analysis['stability_score'] < 0.05
    else:
        analysis['early_stopping_recommended'] = False

    analysis['total_time'] = np.sum(epoch_times)
    analysis['avg_epoch_time'] = np.mean(epoch_times)
    analysis['time_per_loss_reduction'] = None

    if analysis['convergence_rate'] is not None and analysis['convergence_rate'] > 0:
        analysis['time_per_loss_reduction'] = analysis['avg_epoch_time'] / analysis['convergence_rate']

    return analysis
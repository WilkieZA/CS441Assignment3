import numpy as np
from pathlib import Path
import pickle
import time
from collections import namedtuple
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error

IterativeResults = namedtuple('IterativeResults', [
    'best_params', 'best_score', 'best_std', 'all_results', 'tuning_time', 'convergence_history'
])

class IterativeHyperparameterTuner:

    def __init__(self, random_seed=42, cv_folds=5, max_epochs=500, patience=20, tolerance=1e-5):
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        np.random.seed(random_seed)

    def _is_classification(self, y):
        unique_vals = np.unique(y)
        return len(unique_vals) < 20 and np.all(unique_vals == unique_vals.astype(int))

    def _get_cv_splitter(self, y):
        if self._is_classification(y):
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        else:
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

    def _evaluate_configuration(self, network_class, algorithm_class, params, X, y, task_type):
        cv_scores = []
        cv_splitter = self._get_cv_splitter(y)

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                if task_type == 'classification':
                    n_classes = len(np.unique(y))
                    if n_classes == 2:
                        network = network_class(
                            input_dim=X.shape[1],
                            hidden_dim=params['hidden_dim'],
                            output_dim=1,
                            hidden_activation='relu',
                            output_activation='sigmoid',
                            loss='binary_crossentropy'
                        )
                    else:
                        network = network_class(
                            input_dim=X.shape[1],
                            hidden_dim=params['hidden_dim'],
                            output_dim=n_classes,
                            hidden_activation='relu',
                            output_activation='softmax',
                            loss='categorical_crossentropy'
                        )
                else:
                    network = network_class(
                        input_dim=X.shape[1],
                        hidden_dim=params['hidden_dim'],
                        output_dim=1,
                        hidden_activation='relu',
                        output_activation='linear',
                        loss='mse'
                    )

                alg_params = {k: v for k, v in params.items() if k != 'hidden_dim'}
                algorithm = algorithm_class(**alg_params)

                best_val_loss = float('inf')
                patience_counter = 0

                for epoch in range(self.max_epochs):
                    train_loss = algorithm.step(network, X_train, y_train)

                    if task_type == 'classification':
                        val_pred = network.predict_classes(X_val)
                        val_score = accuracy_score(y_val, val_pred)
                        val_loss = -val_score  # Convert to loss for early stopping
                    else:
                        val_pred = network.forward(X_val)
                        val_score = mean_squared_error(y_val, val_pred)
                        val_loss = val_score

                    if val_loss < best_val_loss - self.tolerance:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            break

                if task_type == 'classification':
                    final_pred = network.predict_classes(X_val)
                    score = accuracy_score(y_val, final_pred)
                else:
                    final_pred = network.forward(X_val)
                    score = -mean_squared_error(y_val, final_pred)  # Negative MSE for maximization

                cv_scores.append(score)

            except Exception as e:
                print(f"Error in evaluation: {e}")
                cv_scores.append(0.0 if task_type == 'classification' else -1e6)

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        return mean_score, std_score

    def stage1_coarse_parameter_tuning(self, network_class, algorithm_class, param_grid, X, y, task_type='classification', fixed_h=20):
        print(f"Stage 1: Coarse parameter tuning (h={fixed_h})")

        start_time = time.time()
        results = []
        best_score = -float('inf')
        best_params = None

        param_names = list(param_grid.keys())
        param_combinations = []

        def generate_combinations(param_dict, names, current_combo, index):
            if index == len(names):
                param_combinations.append(current_combo.copy())
                return

            name = names[index]
            for value in param_dict[name]:
                current_combo[name] = value
                generate_combinations(param_dict, names, current_combo, index + 1)

        generate_combinations(param_grid, param_names, {}, 0)

        total_combinations = len(param_combinations)
        print(f"Evaluating {total_combinations} parameter combinations...")

        for i, params in enumerate(param_combinations):
            params['hidden_dim'] = fixed_h  # Fixed architecture

            mean_score, std_score = self._evaluate_configuration(
                network_class, algorithm_class, params, X, y, task_type
            )

            result = {**params, 'mean_score': mean_score, 'std_score': std_score}
            results.append(result)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()

            if (i + 1) % max(1, total_combinations // 10) == 0:
                print(f"  Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%)")

        tuning_time = time.time() - start_time

        return IterativeResults(
            best_params=best_params,
            best_score=best_score,
            best_std=next(r['std_score'] for r in results if r['mean_score'] == best_score),
            all_results=results,
            tuning_time=tuning_time,
            convergence_history=[r['mean_score'] for r in results]
        )

    def stage2_coarse_architecture_search(self, network_class, algorithm_class, best_params, X, y, task_type='classification'):
        print("Stage 2: Coarse architecture search")

        start_time = time.time()
        results = []
        best_score = -float('inf')
        best_h = None

        hidden_dims = [10, 20, 30, 50, 70, 100]

        print(f"Testing {len(hidden_dims)} architectures: {hidden_dims}")

        for i, h in enumerate(hidden_dims):
            params = best_params.copy()
            params['hidden_dim'] = h

            mean_score, std_score = self._evaluate_configuration(
                network_class, algorithm_class, params, X, y, task_type
            )

            result = {'hidden_dim': h, 'mean_score': mean_score, 'std_score': std_score}
            results.append(result)

            if mean_score > best_score:
                best_score = mean_score
                best_h = h

            print(f"  h={h:3d}: {mean_score:.4f} ± {std_score:.4f}")

        best_params_updated = best_params.copy()
        best_params_updated['hidden_dim'] = best_h

        tuning_time = time.time() - start_time

        return IterativeResults(
            best_params=best_params_updated,
            best_score=best_score,
            best_std=next(r['std_score'] for r in results if r['hidden_dim'] == best_h),
            all_results=results,
            tuning_time=tuning_time,
            convergence_history=[r['mean_score'] for r in results]
        )

    def stage3_fine_parameter_tuning(self, network_class, algorithm_class, coarse_best_params, param_grid, X, y, task_type='classification'):
        print(f"Stage 3: Fine parameter tuning (h={coarse_best_params['hidden_dim']})")

        start_time = time.time()
        results = []
        best_score = -float('inf')
        best_params = None

        refined_grid = {}
        for param_name, values in param_grid.items():
            if param_name in coarse_best_params:
                best_val = coarse_best_params[param_name]

                try:
                    best_idx = values.index(best_val)

                    if len(values) == 3 and best_idx == 1:  # Middle value was best
                        if param_name in ['lr']:
                            refined_grid[param_name] = [best_val * 0.3, best_val * 0.7, best_val, best_val * 1.5, best_val * 3.0]
                        elif param_name in ['sigma0', 'lambd']:
                            refined_grid[param_name] = [best_val * 0.1, best_val * 0.5, best_val, best_val * 5.0, best_val * 10.0]
                        else:
                            refined_grid[param_name] = [best_val]
                    else:
                        refined_grid[param_name] = values

                except ValueError:
                    refined_grid[param_name] = values
            else:
                refined_grid[param_name] = values

        param_names = list(refined_grid.keys())
        param_combinations = []

        def generate_combinations(param_dict, names, current_combo, index):
            if index == len(names):
                param_combinations.append(current_combo.copy())
                return

            name = names[index]
            for value in param_dict[name]:
                current_combo[name] = value
                generate_combinations(param_dict, names, current_combo, index + 1)

        generate_combinations(refined_grid, param_names, {}, 0)

        total_combinations = len(param_combinations)
        print(f"Evaluating {total_combinations} refined parameter combinations...")

        for i, params in enumerate(param_combinations):
            params['hidden_dim'] = coarse_best_params['hidden_dim']  # Fixed optimal architecture

            mean_score, std_score = self._evaluate_configuration(
                network_class, algorithm_class, params, X, y, task_type
            )

            result = {**params, 'mean_score': mean_score, 'std_score': std_score}
            results.append(result)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()

            if (i + 1) % max(1, total_combinations // 10) == 0:
                print(f"  Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%)")

        tuning_time = time.time() - start_time

        return IterativeResults(
            best_params=best_params,
            best_score=best_score,
            best_std=next(r['std_score'] for r in results if r['mean_score'] == best_score),
            all_results=results,
            tuning_time=tuning_time,
            convergence_history=[r['mean_score'] for r in results]
        )

    def stage4_fine_architecture_search(self, network_class, algorithm_class, best_params, X, y, task_type='classification'):
        print("Stage 4: Fine architecture search")

        start_time = time.time()
        results = []
        best_score = -float('inf')
        best_h = None

        coarse_h = best_params['hidden_dim']

        if coarse_h <= 20:
            hidden_dims = list(range(max(5, coarse_h - 10), coarse_h + 21, 5))
        elif coarse_h <= 50:
            hidden_dims = list(range(max(10, coarse_h - 20), coarse_h + 31, 10))
        else:
            hidden_dims = list(range(max(20, coarse_h - 30), coarse_h + 41, 20))

        if coarse_h not in hidden_dims:
            hidden_dims.append(coarse_h)
        hidden_dims = sorted(list(set(hidden_dims)))

        print(f"Fine architecture search around h={coarse_h}: {hidden_dims}")

        for i, h in enumerate(hidden_dims):
            params = best_params.copy()
            params['hidden_dim'] = h

            mean_score, std_score = self._evaluate_configuration(
                network_class, algorithm_class, params, X, y, task_type
            )

            result = {'hidden_dim': h, 'mean_score': mean_score, 'std_score': std_score}
            results.append(result)

            if mean_score > best_score:
                best_score = mean_score
                best_h = h

            marker = "🎯" if h == coarse_h else "  "
            print(f"  {marker} h={h:3d}: {mean_score:.4f} ± {std_score:.4f}")

        final_params = best_params.copy()
        final_params['hidden_dim'] = best_h

        tuning_time = time.time() - start_time

        return IterativeResults(
            best_params=final_params,
            best_score=best_score,
            best_std=next(r['std_score'] for r in results if r['hidden_dim'] == best_h),
            all_results=results,
            tuning_time=tuning_time,
            convergence_history=[r['mean_score'] for r in results]
        )

    def iterative_tuning_pipeline(self, network_class, algorithm_configs, X, y, dataset_name):

        print(f"\n{'='*80}")
        print(f"ITERATIVE HYPERPARAMETER TUNING: {dataset_name.upper()}")
        print(f"{'='*80}")

        task_type = 'classification' if self._is_classification(y) else 'regression'
        print(f"Task type: {task_type}")
        print(f"Dataset shape: {X.shape}, Classes/Outputs: {len(np.unique(y))}")

        from config.settings import ALGORITHM_PARAMS

        all_results = {}

        for alg_name, algorithm_class in algorithm_configs.items():
            print(f"\n{'-'*60}")
            print(f"ALGORITHM: {alg_name.upper()}")
            print(f"{'-'*60}")

            total_start_time = time.time()
            param_grid = ALGORITHM_PARAMS[alg_name]

            stage1 = self.stage1_coarse_parameter_tuning(
                network_class, algorithm_class, param_grid, X, y, task_type
            )
            print(f"Stage 1 best: {stage1.best_params} -> {stage1.best_score:.4f} ± {stage1.best_std:.4f}")

            stage2 = self.stage2_coarse_architecture_search(
                network_class, algorithm_class,
                {k: v for k, v in stage1.best_params.items() if k != 'hidden_dim'},
                X, y, task_type
            )
            print(f"Stage 2 best: h={stage2.best_params['hidden_dim']} -> {stage2.best_score:.4f} ± {stage2.best_std:.4f}")

            stage3 = self.stage3_fine_parameter_tuning(
                network_class, algorithm_class, stage2.best_params, param_grid, X, y, task_type
            )
            print(f"Stage 3 best: {stage3.best_params} -> {stage3.best_score:.4f} ± {stage3.best_std:.4f}")

            stage4 = self.stage4_fine_architecture_search(
                network_class, algorithm_class, stage3.best_params, X, y, task_type
            )
            print(f"Stage 4 final: {stage4.best_params} -> {stage4.best_score:.4f} ± {stage4.best_std:.4f}")

            total_time = time.time() - total_start_time

            all_results[alg_name] = {
                'stage1': stage1,
                'stage2': stage2,
                'stage3': stage3,
                'stage4': stage4,
                'final_params': stage4.best_params,
                'final_score': stage4.best_score,
                'final_std': stage4.best_std,
                'total_time': total_time
            }

            print(f"Total time: {total_time:.1f}s")

        results_dir = Path("results/iterative_tuning")
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"{dataset_name}_iterative_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)

        print(f"\nResults saved to: {results_file}")

        return all_results
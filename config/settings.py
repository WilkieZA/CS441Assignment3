import numpy as np

RANDOM_SEED = 42

DEFAULT_HIDDEN_LAYERS = [10, 20, 30, 50]
ACTIVATION_FUNCTIONS = ['sigmoid', 'tanh', 'relu']
OUTPUT_ACTIVATIONS = {
    'binary_classification': 'sigmoid',
    'multiclass_classification': 'softmax',
    'regression': 'linear'
}

LOSS_FUNCTIONS = {
    'binary_classification': 'binary_crossentropy',
    'multiclass_classification': 'categorical_crossentropy',
    'regression': 'mse'
}

SGD_CONFIG = {
    'lr': [0.001, 0.01, 0.1],        # Learning rate
    'momentum': [0.0, 0.5, 0.9]      # Momentum coefficient
}

SCG_CONFIG = {
    'sigma0': [1e-6, 1e-5, 1e-4],    # Hessian approximation σ
    'lambd': [1e-6, 1e-4, 1e-2]     # Levenberg-Marquardt
}

LEAPFROG_CONFIG = {
    'dt': [0.01, 0.1, 0.5],          # Initial time step
    'delta_max': [0.1, 1.0, 2.0],    # Maximum step constraint
    'xi': [0.001, 0.01, 0.05],       # Growth factor
    'm': [3, 5, 7]                   # Restart threshold
}

ALGORITHM_PARAMS = {
    'sgd': SGD_CONFIG,
    'scg': SCG_CONFIG,
    'leapfrog': LEAPFROG_CONFIG
}

MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 50
VALIDATION_SPLIT = 0.2
NUM_RUNS = 10

DATA_PATHS = {
    'raw': 'data/raw/',
    'processed': 'data/processed/',
    'synthetic': 'data/synthetic/'
}

RESULTS_PATHS = {
    'plots': 'results/plots/',
    'data': 'results/data/',
    'statistics': 'results/statistics/'
}
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

def generate_polynomial_dataset(n_samples=500, noise_std=0.1):
    print("1. Generating Polynomial Dataset...")

    X = np.random.uniform(-2, 3, n_samples).reshape(-1, 1)

    y_true = X.ravel()**3 - 2*X.ravel()**2 + X.ravel()

    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    print(f"   Generated {n_samples} samples")
    print(f"   Input range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Output range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Noise std: {noise_std}")

    return X, y, y_true

def generate_sinusoidal_dataset(n_samples=800, noise_std=0.15):
    print("\n2. Generating Sinusoidal Dataset...")

    X1 = np.random.uniform(-1, 1, n_samples)
    X2 = np.random.uniform(-1, 1, n_samples)
    X = np.column_stack([X1, X2])

    y_true = np.sin(2 * np.pi * X1) * np.cos(2 * np.pi * X2)

    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    print(f"   Generated {n_samples} samples")
    print(f"   Input ranges: X1[{X1.min():.2f}, {X1.max():.2f}], X2[{X2.min():.2f}, {X2.max():.2f}]")
    print(f"   Output range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Noise std: {noise_std}")

    return X, y, y_true

def generate_gaussian_mixture_dataset(n_samples=1000, noise_std=0.2):
    print("\n3. Generating Gaussian Mixture Dataset...")

    X1 = np.random.uniform(-3, 3, n_samples)
    X2 = np.random.uniform(-3, 3, n_samples)
    X3 = np.random.uniform(-3, 3, n_samples)
    X = np.column_stack([X1, X2, X3])

    centers = np.array([[-1, -1, -1], [1, 1, 0], [0, -1, 2]])
    amplitudes = np.array([2.0, 1.5, 1.8])
    sigmas = np.array([1.5, 1.2, 1.0])

    y_true = np.zeros(n_samples)
    for i in range(3):
        distances_sq = np.sum((X - centers[i])**2, axis=1)
        y_true += amplitudes[i] * np.exp(-distances_sq / (2 * sigmas[i]**2))

    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    print(f"   Generated {n_samples} samples")
    print(f"   Input ranges: X1[{X1.min():.2f}, {X1.max():.2f}], X2[{X2.min():.2f}, {X2.max():.2f}], X3[{X3.min():.2f}, {X3.max():.2f}]")
    print(f"   Output range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Noise std: {noise_std}")
    print(f"   Gaussian centers: {centers}")
    print(f"   Amplitudes: {amplitudes}")
    print(f"   Sigmas: {sigmas}")

    return X, y, y_true

def create_visualizations():
    print("\n4. Creating Visualizations...")

    Path("results/plots").mkdir(parents=True, exist_ok=True)

    poly_X = np.load('../data/synthetic/polynomial_X.npy')
    poly_y = np.load('../data/synthetic/polynomial_y.npy')

    sin_X = np.load('../data/synthetic/sinusoidal_X.npy')
    sin_y = np.load('../data/synthetic/sinusoidal_y.npy')

    gauss_X = np.load('../data/synthetic/gaussian_mixture_X.npy')
    gauss_y = np.load('../data/synthetic/gaussian_mixture_y.npy')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].scatter(poly_X, poly_y, alpha=0.6, s=20)
    axes[0, 0].set_title('Polynomial Function: $f(x) = x^3 - 2x^2 + x + \\epsilon$')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True, alpha=0.3)

    x1_grid = np.linspace(-1, 1, 50)
    x2_grid = np.linspace(-1, 1, 50)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
    Y_mesh = np.sin(2 * np.pi * X1_mesh) * np.cos(2 * np.pi * X2_mesh)

    contour = axes[0, 1].contourf(X1_mesh, X2_mesh, Y_mesh, levels=20, cmap='viridis')
    axes[0, 1].scatter(sin_X[:200, 0], sin_X[:200, 1], c=sin_y[:200], s=20, alpha=0.7, cmap='viridis')
    axes[0, 1].set_title('Sinusoidal Function: $f(x,y) = \\sin(2\\pi x)\\cos(2\\pi y) + \\epsilon$')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(contour, ax=axes[0, 1])

    axes[1, 0].scatter(gauss_X[:, 0], gauss_X[:, 1], c=gauss_y, s=20, alpha=0.7, cmap='plasma')
    axes[1, 0].set_title('Gaussian Mixture Surface (X1-X2 projection)')
    axes[1, 0].set_xlabel('X1')
    axes[1, 0].set_ylabel('X2')

    complexities = ['Low\n(Polynomial)', 'Medium\n(Sinusoidal)', 'High\n(Gaussian Mix)']
    dimensions = [1, 2, 3]
    sample_sizes = [len(poly_y), len(sin_y), len(gauss_y)]

    axes[1, 1].bar(complexities, dimensions, alpha=0.7, color=['blue', 'green', 'red'])
    axes[1, 1].set_title('Dataset Complexity Progression')
    axes[1, 1].set_ylabel('Input Dimensions')
    axes[1, 1].grid(True, alpha=0.3)

    for i, (complexity, size) in enumerate(zip(complexities, sample_sizes)):
        axes[1, 1].text(i, dimensions[i] + 0.1, f'n={size}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('../results/plots/function_approximation_datasets.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(" Visualization saved to results/plots/function_approximation_datasets.png")

def main():
    print("=== GENERATING FUNCTION APPROXIMATION DATASETS ===")
    print("=" * 60)

    Path("data/synthetic").mkdir(parents=True, exist_ok=True)

    poly_X, poly_y, poly_y_true = generate_polynomial_dataset()
    sin_X, sin_y, sin_y_true = generate_sinusoidal_dataset()
    gauss_X, gauss_y, gauss_y_true = generate_gaussian_mixture_dataset()

    from sklearn.preprocessing import StandardScaler

    poly_scaler = StandardScaler()
    poly_X_scaled = poly_scaler.fit_transform(poly_X)

    sin_scaler = StandardScaler()
    sin_X_scaled = sin_scaler.fit_transform(sin_X)

    gauss_scaler = StandardScaler()
    gauss_X_scaled = gauss_scaler.fit_transform(gauss_X)

    print("\n5. Standardizing and Saving Datasets...")

    np.save('../data/synthetic/polynomial_X.npy', poly_X_scaled)
    np.save('../data/synthetic/polynomial_y.npy', poly_y)

    np.save('../data/synthetic/sinusoidal_X.npy', sin_X_scaled)
    np.save('../data/synthetic/sinusoidal_y.npy', sin_y)

    np.save('../data/synthetic/gaussian_mixture_X.npy', gauss_X_scaled)
    np.save('../data/synthetic/gaussian_mixture_y.npy', gauss_y)

    import pickle
    with open('../data/synthetic/polynomial_scaler.pkl', 'wb') as f:
        pickle.dump(poly_scaler, f)
    with open('../data/synthetic/sinusoidal_scaler.pkl', 'wb') as f:
        pickle.dump(sin_scaler, f)
    with open('../data/synthetic/gaussian_mixture_scaler.pkl', 'wb') as f:
        pickle.dump(gauss_scaler, f)

    print(f"    Polynomial: {poly_X_scaled.shape} features, {poly_y.shape} targets")
    print(f"    Sinusoidal: {sin_X_scaled.shape} features, {sin_y.shape} targets")
    print(f"    Gaussian Mixture: {gauss_X_scaled.shape} features, {gauss_y.shape} targets")

    create_visualizations()

    print("\n" + "=" * 60)
    print("✅ ALL FUNCTION APPROXIMATION DATASETS GENERATED!")
    print("Ready for neural network regression experiments")
    print("\nDataset Summary:")
    print(f"• Polynomial (1D): {len(poly_y)} samples - Low complexity")
    print(f"• Sinusoidal (2D): {len(sin_y)} samples - Medium complexity")
    print(f"• Gaussian Mix (3D): {len(gauss_y)} samples - High complexity")

if __name__ == "__main__":
    main()
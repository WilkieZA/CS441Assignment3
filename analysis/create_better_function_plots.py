
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

np.random.seed(42)

Path("results/plots").mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 2, 1)

x_poly = np.linspace(-2, 3, 200)
y_poly_true = x_poly**3 - 2*x_poly**2 + x_poly

ax1.plot(x_poly, y_poly_true, 'b-', linewidth=2, label='True function', zorder=2)

np.random.seed(42)
x_samples = np.random.uniform(-2, 3, 80)
y_samples = x_samples**3 - 2*x_samples**2 + x_samples + np.random.normal(0, 0.1, 80)
ax1.scatter(x_samples, y_samples, c='red', s=20, alpha=0.6, label='Training samples', zorder=3)

ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$f(x)$', fontsize=12)
ax1.set_title('Polynomial: $f(x) = x^3 - 2x^2 + x + \\epsilon$', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')

x1_mesh = np.linspace(-1, 1, 50)
x2_mesh = np.linspace(-1, 1, 50)
X1, X2 = np.meshgrid(x1_mesh, x2_mesh)
Y_surface = np.sin(2 * np.pi * X1) * np.cos(2 * np.pi * X2)

surf = ax2.plot_surface(X1, X2, Y_surface, cmap='viridis', alpha=0.8,
                        linewidth=0, antialiased=True, shade=True)

np.random.seed(42)
n_samples = 40
x1_samples = np.random.uniform(-1, 1, n_samples)
x2_samples = np.random.uniform(-1, 1, n_samples)
y_samples = np.sin(2*np.pi*x1_samples) * np.cos(2*np.pi*x2_samples) + np.random.normal(0, 0.1, n_samples)
ax2.scatter(x1_samples, x2_samples, y_samples, c='red', s=20, alpha=0.8, depthshade=True)

ax2.set_xlabel('$x$', fontsize=11)
ax2.set_ylabel('$y$', fontsize=11)
ax2.set_zlabel('$f(x,y)$', fontsize=11)
ax2.set_title('Sinusoidal: $f(x,y) = \\sin(2\\pi x)\\cos(2\\pi y) + \\epsilon$',
              fontsize=13, fontweight='bold', pad=15)

fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('results/plots/function_approximation_clean.png', dpi=300, bbox_inches='tight')
print("Saved: results/plots/function_approximation_clean.png")

plt.show()

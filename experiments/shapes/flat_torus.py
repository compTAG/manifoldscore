"""
flat_torus.py

Utilities for generating samples of points on the 2D flat torus (unit square with periodic boundary conditions)
for manifoldscore experiments.

Supports:
- Regular grid sampling.
- Regular grid with jitter.
- Uniform random sampling.
- Cross-stratified and noisy-cross structures.
- Single point sample.

Example usage:
--------------
from shapes.flat_torus import generate_flat_torus_sample, plot_torus_point_cloud

# Generate 1000 uniform points on the flat torus
torus_sample = generate_flat_torus_sample(n=1000, sample_type="unif")

# Visualize 2D projection
plot_torus_point_cloud(torus_sample.point_cloud.numpy())
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from manifoldscore.utilities import ManifoldSample

# ------------------------------------------------------------
# --- Toroidal utilities
# ------------------------------------------------------------

def toroidal_distance_matrix(points):
    """
    Compute pairwise toroidal distances between 2D points on the unit square.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with coordinates in [0, 1]^2.

    Returns
    -------
    np.ndarray
        Distance matrix (N x N) with toroidal (wrapped) Euclidean distances.
    """
    N = len(points)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        diff = np.abs(points[i] - points)
        diff = np.minimum(diff, 1.0 - diff)  # wrap-around distance
        dist_matrix[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    return dist_matrix


# ------------------------------------------------------------
# --- Sampling functions
# ------------------------------------------------------------

def generate_unit_square(n):
    """Generate n uniform random points in [0, 1]^2."""
    return np.random.rand(n, 2)

def generate_cross_stratification(n):
    """Generate a cross-shaped distribution (vertical + horizontal line)."""
    n_half = n // 2
    x1 = np.linspace(0, 1, n_half)
    y1 = np.full_like(x1, 0.5)
    y2 = np.linspace(0, 1, n - n_half)
    x2 = np.full_like(y2, 0.5)
    return np.vstack([np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1)])

def generate_cross_with_noise(n, noise_level=0.02):
    """Generate a cross distribution with Gaussian noise."""
    X = generate_cross_stratification(n)
    X += np.random.normal(0, noise_level, X.shape)
    X %= 1.0  # wrap around
    return X


# ------------------------------------------------------------
# --- Main torus sampler
# ------------------------------------------------------------

def generate_flat_torus_sample(n=1000, sample_type="unif") -> ManifoldSample:
    """
    Generate a sample of n points on the flat torus (unit square with wrap-around).

    Parameters
    ----------
    n : int
        Number of sample points.
    sample_type : str
        Type of sampling:
        - 'regular'
        - 'regular_jitter'
        - 'unif'
        - 'cross'
        - 'noisycross'
        - 'point'

    Returns
    -------
    ManifoldSample
        The sampled manifold with precomputed toroidal distance matrix.
    """
    sample_type = sample_type.lower()

    if sample_type == "regular":
        side = int(np.ceil(np.sqrt(n)))
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        X = np.column_stack((grid_x.ravel(), grid_y.ravel()))[:n]

    elif sample_type == "regular_jitter":
        side = int(np.ceil(np.sqrt(n)))
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        X = np.column_stack((grid_x.ravel(), grid_y.ravel()))[:n]
        jitter = np.random.uniform(-0.5 / side, 0.5 / side, X.shape)
        X = np.clip(X + jitter, 0, 1)

    elif sample_type == "unif":
        X = generate_unit_square(n)

    elif sample_type == "cross":
        X = generate_cross_stratification(n)

    elif sample_type == "noisycross":
        X = generate_cross_with_noise(n)

    elif sample_type == "point":
        X = np.array([[0.5, 0.5]] * n)

    else:
        raise ValueError(f"Unsupported sample_type '{sample_type}'")

    dist_matrix = toroidal_distance_matrix(X)
    s = dist_matrix.max() if dist_matrix.max() != 0 else 1.0
    dist_matrix_norm = dist_matrix / s

    point_cloud = torch.tensor(X, dtype=torch.float32)
    dist_tensor = torch.tensor(dist_matrix_norm, dtype=torch.float32)
    manifold_sample = ManifoldSample(point_cloud, distance_matrix=dist_tensor)
    return manifold_sample


# ------------------------------------------------------------
# --- Visualization helper
# ------------------------------------------------------------

def plot_torus_point_cloud(points, s=10, title="Flat Torus Sample (unit square)"):
    """Simple 2D scatter plot of torus samples."""
    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], s=s, color="royalblue", alpha=0.7)
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


# ------------------------------------------------------------
# --- Test / Example run
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Generating uniform points on the flat torus...")
    sample = generate_flat_torus_sample(n=1000, sample_type="unif")
    plot_torus_point_cloud(sample.point_cloud.numpy())

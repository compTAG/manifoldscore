"""
sphere.py

Utilities for generating samples of points on spheres for manifoldscore experiments.

Supports:
- Uniform random sampling on an n-dimensional unit sphere.
- Simple 3D visualization for low-dimensional spheres.

Example usage:
--------------
from sphere_sampling import generate_sphere_sample, plot_3d_point_cloud

# Generate 1000 uniform points on S^2 (the 3D sphere)
points = generate_sphere_sample(n=1000, d=3, sample_type="uniform")

# Visualize if 3D
plot_3d_point_cloud(points)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from manifoldscore.utilities import ManifoldSample


# ------------------------------------------------------------
# ---  Sphere sampling
# ------------------------------------------------------------

def generate_sphere_sample(n=1000, d=3, sample_type="uniform") -> ManifoldSample:
    """
    Generate a sample of n points on the unit sphere in R^d.

    Parameters
    ----------
    n : int
        Number of sample points.
    d : int
        Dimension of the embedding space (sphere is S^{d-1} ⊂ R^d).
    sample_type : str
        Type of sampling:
        - 'uniform': random uniform distribution on the sphere.
        - 'regular': even spacing of points (Fibonacci sphere for d=3).

    Returns
    -------
    np.ndarray
        Array of shape (n, d) containing sampled points.
    """
    sample_type = sample_type.lower()

    if sample_type == "uniform":
        # Draw from isotropic Gaussian and normalize each vector
        S = np.random.normal(0, 1, (n, d))
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        S /= norms

    elif sample_type == "regular":
        if d == 3:
            # Fibonacci sphere algorithm
            indices = np.arange(0, n, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n)
            theta = np.pi * (1 + 5 ** 0.5) * indices  # golden angle

            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            S = np.stack((x, y, z), axis=1)
        else:
            # For d>3, fallback to uniform (true regular grids are complex)
            print(f"[Warning] Regular sampling not implemented for d={d}. Using uniform sampling instead.")
            S = np.random.normal(0, 1, (n, d))
            norms = np.linalg.norm(S, axis=1, keepdims=True)
            S /= norms

    elif sample_type == "point":
        # n copies of the north pole, perturbed slightly to avoid zero distances
        S = np.zeros((n, d))
        S[:, 0] = 1.0  # All points at (1, 0, 0, ..., 0)
        S += 1e-6 * np.random.randn(n, d)  # Small random perturbation

    elif sample_type == "poles":
        if d == 3:
            # North pole, south pole, east pole, west pole, with 1/4 of points at each
            n_quarter = n // 4
            S = np.zeros((n, 3))
            S[:n_quarter, :] = np.array([1, 0, 0])  # North pole
            S[n_quarter:2*n_quarter, :] = np.array([-1, 0, 0])  # South pole
            S[2*n_quarter:3*n_quarter, :] = np.array([0, 1, 0])  # East pole
            S[3*n_quarter:, :] = np.array([0, -1, 0])  # West pole
            S += 1e-6 * np.random.randn(n, 3)  # Small random perturbation
        else:
            raise ValueError("Pole sampling is only implemented for d=3.")

    elif sample_type == "cross_subset":
        if d == 3:
            # Split points evenly between x=0 plane and y=0 plane
            n_half = n // 2
            # x=0 plane: sample points on the circle y^2 + z^2 = 1
            theta1 = np.random.uniform(0, 2*np.pi, n_half)
            x1 = np.zeros(n_half)
            y1 = np.cos(theta1)
            z1 = np.sin(theta1)
            # y=0 plane: sample points on the circle x^2 + z^2 = 1
            theta2 = np.random.uniform(0, 2*np.pi, n - n_half)
            x2 = np.cos(theta2)
            y2 = np.zeros(n - n_half)
            z2 = np.sin(theta2)
            # Combine
            S = np.vstack([np.stack([x1, y1, z1], axis=1),
                        np.stack([x2, y2, z2], axis=1)])
        else:
            raise ValueError("cross_subset sampling is only implemented for d=3.")

    else:
        raise ValueError(f"Unsupported sample_type '{sample_type}'. Supported: 'uniform', 'regular'.")

    # Create ManifoldSample instance (distance matrix will be computed)
    manifold_sample = ManifoldSample(point_cloud=torch.tensor(S, dtype=torch.float32))
    return manifold_sample


# ------------------------------------------------------------
# ---  3D plotting helpers
# ------------------------------------------------------------

def plot_3d_point_cloud(A, s=10, title="3D Sphere Sample"):
    """
    Plot a 3D scatter of a point cloud (only works if A has shape (N, 3)).

    Parameters
    ----------
    A : np.ndarray
        Array of shape (N, 3)
    s : float
        Marker size
    title : str
        Title for the plot
    """
    if A.shape[1] != 3:
        raise ValueError("plot_3d_point_cloud() requires 3D data (shape (N, 3)).")

    X, Y, Z = A[:, 0], A[:, 1], A[:, 2]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=s, alpha=0.7, color='royalblue')

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    # remove pane backgrounds
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    plt.axis("off")
    plt.show()


def plot_3d_mesh(X, Y, Z, color="lightgray", alpha=0.3, title="3D Surface Mesh"):
    """
    Plot a 3D mesh (useful for showing surface geometry).

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Meshgrid coordinates
    color : str
        Surface color
    alpha : float
        Transparency
    title : str
        Title of the plot
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, rstride=1, cstride=1, edgecolor='none')
    ax.set_title(title)
    plt.show()


# ------------------------------------------------------------
# ---  Mesh for visualization (optional)
# ------------------------------------------------------------

def generate_sphere_mesh(steps=50, radius=1.0):
    """
    Generate a 3D sphere mesh (useful for visualization).

    Parameters
    ----------
    steps : int
        Resolution of the mesh grid
    radius : float
        Sphere radius

    Returns
    -------
    X, Y, Z : np.ndarray
        Meshgrid arrays defining the sphere surface
    """
    phi = np.linspace(0, np.pi, steps)
    theta = np.linspace(0, 2 * np.pi, steps)
    phi, theta = np.meshgrid(phi, theta)
    X = radius * np.sin(phi) * np.cos(theta)
    Y = radius * np.sin(phi) * np.sin(theta)
    Z = radius * np.cos(phi)
    return X, Y, Z


# ------------------------------------------------------------
# ---  Test / Example run
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    print("Generating uniform points on S^2...")
    points = generate_sphere_sample(n=1000, d=3, sample_type="cross_subset").point_cloud.numpy()

    # # Visualize
    plot_3d_point_cloud(points, s=5, title="Sample on S²")

    # # Optional: plot sphere surface
    # X, Y, Z = generate_sphere_mesh(steps=50)
    # plot_3d_mesh(X, Y, Z, color="lightgray", alpha=0.6)

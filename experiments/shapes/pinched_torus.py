"""
pinched_torus.py

Utilities for loading and visualizing a fixed sample of points on a
pinched torus for manifoldscore experiments.

The data is loaded from a file, subsampled, and presented as a ManifoldSample.

Example usage:
--------------
from pinched_torus import generate_pinched_torus_sample, plot_3d_point_cloud

# Generate a sample of 1000 points on the pinched torus
points_sample = generate_pinched_torus_sample(n=1000)

# Visualize if 3D
plot_3d_point_cloud(points_sample.point_cloud.numpy())
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# NOTE: Assuming ManifoldSample is available, as in the original sphere.py
from manifoldscore.utilities import ManifoldSample


# ------------------------------------------------------------
# --- Pinched Torus sampling
# ------------------------------------------------------------

def generate_pinched_torus_sample(n=1000) -> ManifoldSample:
    """
    Generate a sample of n points on the pinched torus from a fixed dataset file.

    Parameters
    ----------
    n : int
        Number of sample points to subsample.

    Returns
    -------
    ManifoldSample
        ManifoldSample instance containing the subsampled points.
    """
    # Load the data from the text file (assumes file is in 'data/' directory)
    # This must be adjusted if the file path is different in the actual environment
    try:
        data = np.loadtxt('shapes/data/Pinched_torus.txt')
    except FileNotFoundError:
        raise FileNotFoundError("[Error] Pinched_torus.txt not found. Please ensure it is in the 'data/' directory and unzipped.")
        
    # Subsample random n points
    np.random.seed(42)
    if data.shape[0] > n:
        indices = np.random.choice(data.shape[0], size=n, replace=False)
        subsampled_data = data[indices, :]
    else:
        # If the dataset has fewer points than requested n, use all data and warn
        if data.shape[0] < n:
            print(f"[Warning] Requested {n} points, but only {data.shape[0]} available. Using all data.")
        subsampled_data = data

    # Create ManifoldSample instance
    manifold_sample = ManifoldSample(point_cloud=torch.tensor(subsampled_data, dtype=torch.float32))
    return manifold_sample


# ------------------------------------------------------------
# --- 3D plotting helpers (Adapted from the sphere.py example)
# ------------------------------------------------------------

def plot_3d_point_cloud(A, s=5, title="3D Pinched Torus Sample"):
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

    # Set axis limits based on the context example for consistent view
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-3, 3)

    # remove pane backgrounds and grid lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# --- Test / Example run
# ------------------------------------------------------------

if __name__ == "__main__":

    print("Generating sample of points on the pinched torus...")
    # Generate 1000 points
    try:
        points_sample = generate_pinched_torus_sample(n=1000)
        points = points_sample.point_cloud.numpy()

        # Visualize
        plot_3d_point_cloud(points, s=5, title="Sample on Pinched Torus")
        
    except ValueError as e:
        print(f"Skipping visualization due to error: {e}")
    except NameError:
        print("Skipping visualization: ManifoldSample class not defined or imported correctly.")

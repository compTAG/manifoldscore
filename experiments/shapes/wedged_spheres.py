"""
wedged_spheres.py

Utilities for loading and visualizing a fixed sample of points on
wedged spheres for manifoldscore experiments.

The data is loaded from a file, subsampled, and presented as a ManifoldSample.

Example usage:
--------------
from wedged_spheres import generate_wedged_spheres_sample, plot_3d_point_cloud

# Generate a sample of 1000 points on the wedged spheres
points_sample = generate_wedged_spheres_sample(n=1000)

# Visualize if 3D
plot_3d_point_cloud(points_sample.point_cloud.numpy())
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# NOTE: Assuming ManifoldSample is available, as in the original sphere.py
from manifoldscore.utilities import ManifoldSample


# ------------------------------------------------------------
# --- Wedged Spheres sampling
# ------------------------------------------------------------

def generate_wedged_spheres_sample(n=1000) -> ManifoldSample:
    """
    Generate a sample of n points on the wedged spheres from a fixed dataset file.

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
        data = np.loadtxt('shapes/data/Wedged_spheres_2D.txt')
    except FileNotFoundError:
        raise FileNotFoundError("[Error] Wedged_spheres_2D.txt not found. Please ensure it is in the 'data/' directory and unzipped.")

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

def plot_3d_point_cloud(A, s=5, title="3D Wedged Spheres Sample"):
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
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

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

    print("Generating sample of points on the wedged spheres...")
    # Generate 1000 points
    try:
        points_sample = generate_wedged_spheres_sample(n=2000)
        points = points_sample.point_cloud.numpy()

        # Visualize
        plot_3d_point_cloud(points, s=5, title="Sample on Wedged Spheres")
        
    except ValueError as e:
        print(f"Skipping visualization due to error: {e}")
    except NameError:
        print("Skipping visualization: ManifoldSample class not defined or imported correctly.")

"""
local.py

Experiment script for computing and visualizing Manifold Score on the 
fixed Pinched Torus dataset.

Focuses on visualizing the disaggregated score on the 3D point cloud.
"""

import numpy as np
import matplotlib.pyplot as plt

# Assuming these utilities are available in the project structure
from utils import compute_manifold_score 
from shapes.pinched_torus import generate_pinched_torus_sample 

# NOTE: The original pinched_torus context used KFunction, ManifoldValidator, and Isomap.
# This experiment file uses the generic 'compute_manifold_score' function, 
# which is assumed to handle the underlying distance matrix computation 
# (e.g., using Isomap if needed for geodesic distance on the manifold).


# ------------------------------------------------------------
# --- Plotting Helper (Adapted from original context)
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

def plot_pt(data, scores, title="Pinched Torus with Manifold Scores"):
    """
    Plots the 3D Pinched Torus point cloud, colored by manifold scores,
    with formatting similar to the provided example (flattened 3D view, horizontal colorbar).
    """
    if data.shape[1] != 3:
        raise ValueError("plot_pt() requires 3D data (shape (N, 3)).")

    # Figure setup
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-2, 2)

    # Colormap setup
    cmap = plt.cm.hot_r  # yellow → red → dark
    norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    colors = cmap(norm(scores))

    # Scatter plot (thin marker style)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, s=6, alpha=0.9, linewidth=0)

    # Set viewpoint and flatten perspective
    ax.view_init(elev=15, azim=180)
    ax.dist = 10  # zoom control

    # Remove all axis elements for clean look
    ax.set_axis_off()
    ax.grid(False)

    # Transparent panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # Maintain proportion
    ax.set_box_aspect([2, 1, 1])

    # Add horizontal colorbar below
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0, aspect=40, fraction=0.05)
    cbar.set_label('Local Manifold Score', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Tighter layout, no title
    plt.tight_layout()
    plt.show()



# ------------------------------------------------------------
# --- Experiment Logic
# ------------------------------------------------------------

def exp_pinched_torus(num_points, visualize=True):
    """
    Runs the Manifold Score experiment on the Pinched Torus dataset.
    
    Parameters
    ----------
    num_points : int
        Number of points to subsample from the dataset.
    visualize : bool
        If True, plots the result colored by disaggregated scores.

    Returns
    -------
    tuple
        (disaggregated_scores: np.ndarray, aggregated_score: float)
    """
    print(f"Loading and subsampling {num_points} points for Pinched Torus...")
    
    # 1. Load the data using the utility function
    # The utility handles loading and subsampling 'num_points'
    torus_sample = generate_pinched_torus_sample(n=num_points)
    
    # The pinched torus is a 2D manifold embedded in R^3.
    # The theoretical K-function for a 2D manifold is K(r) = πr^2.
    theoretical_func = lambda r: np.pi * (r**2)  
    
    EC = 0  # Euler characteristic for torus
    curvature_correction_func = lambda r: (1.0 - (np.pi * EC / 24)) ** -1
    
    # Parameters a and b for the integration range are taken from the context
    a = 0.0
    b = 0.25
    
    # 2. Compute Manifold Scores
    disagg_scores, agg_score = compute_manifold_score(
        torus_sample,
        theoretical_func,
        disagg_correction=curvature_correction_func,
        agg_correction=curvature_correction_func,
        a=a,
        b=b,
        step=0.01,
        device='cpu',
        # NOTE: Setting visualize=False here prevents intermediate plots from compute_manifold_score
        visualize=False 
    )

    # 3. Visualization
    if visualize and disagg_scores is not None:
        plot_pt(torus_sample.point_cloud.numpy(), disagg_scores, 
                title=f"Pinched Torus Sample (N={num_points}) - Agg Score: {agg_score:.4f}")

    return disagg_scores, agg_score

def main():
    # Set the number of points (as in the original context)
    num_points = 2000
    
    # Run the single experiment
    disagg_scores, agg_score = exp_pinched_torus(
        num_points=num_points, 
        visualize=True
    )
    
    print("-" * 50)
    print(f"Experiment Complete: Pinched Torus (N={num_points})")
    print(f"Aggregated Manifold Score: {agg_score:.6f}")
    if disagg_scores is not None:
        print(f"Max Disaggregated Score: {disagg_scores.max():.6f}")
        print(f"Min Disaggregated Score: {disagg_scores.min():.6f}")

    # Keep the plot window open
    plt.show()

if __name__ == "__main__":
    main()
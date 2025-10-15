import numpy as np
import torch
import matplotlib.pyplot as plt

from manifoldscore.score import ManifoldScore
from manifoldscore.utilities import ManifoldSample

from shapes.flat_torus import (
    generate_unit_square,
    generate_cross_stratification,
    generate_cross_with_noise,
    toroidal_distance_matrix,
    plot_kf_vs_radius,
)

# ========================================================
# =============== Core Manifold Score Logic ===============
# ========================================================

def compute_manifold_score_on_torus(sample: ManifoldSample, theoretical_func, a=0.0, b=0.7, step=0.01, device='cpu', visualize=False):
    """
    Compute the manifold score for a given ManifoldSample of a flat torus.

    Parameters
    ----------
    sample : ManifoldSample
        The manifold sample containing the point cloud and distance matrix.
    theoretical_func : callable
        The theoretical K-function for a 2D uniform manifold.
    a, b : float
        Range of radii to evaluate (from a to b).
    step : float
        Step size for discretizing radii.
    device : str
        'cpu', 'cuda', or 'mps'.
    visualize : bool
        Whether to visualize the empirical vs theoretical K-function.

    Returns
    -------
    disaggregated_scores : torch.Tensor
        The score per point.
    aggregated_score : float
        The overall aggregated manifold score.
    """
    scorer = ManifoldScore(sample, a=a, b=b, step=step, device=device)
    disagg_scores, agg_score = scorer.compute_scores(theoretical_func)

    if visualize:
        plot_kf_vs_radius(scorer, theoretical_func, title="Flat Torus: K-function vs Radius")

    return disagg_scores, agg_score


# ========================================================
# =============== Flat Torus Experiment ==================
# ========================================================

def exp_flat_torus(num_points, sample_type, a=0.0, b=0.7, visualize=False):
    """
    Run a single experiment on the flat torus for a given sample type and size.

    Parameters
    ----------
    num_points : int
        Number of points to sample on the torus.
    sample_type : str
        Sampling type: 'regular', 'unif', 'cross', 'noisycross', or 'point'.
    a, b : float
        Radius range for K-function evaluation.
    visualize : bool
        Whether to visualize K-function comparison.

    Returns
    -------
    float
        Aggregated manifold score.
    """
    # --- Step 1: Sample base points
    if sample_type == "regular":
        side = int(np.ceil(np.sqrt(num_points)))
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        X_base = np.column_stack((grid_x.ravel(), grid_y.ravel()))[:num_points]
    elif sample_type == "unif":
        X_base = generate_unit_square(num_points)
    elif sample_type == "cross":
        X_base = generate_cross_stratification(num_points)
    elif sample_type == "noisycross":
        X_base = generate_cross_with_noise(num_points)
    elif sample_type == "point":
        X_base = np.array([[0.5, 0.5]] * num_points)
    else:
        raise ValueError(f"Unknown sample_type '{sample_type}'")

    # --- Step 2: Compute toroidal distance matrix
    dist_matrix = toroidal_distance_matrix(X_base)
    s = dist_matrix.max() if dist_matrix.max() != 0 else 1.0
    dist_matrix_norm = dist_matrix / s

    # --- Step 3: Wrap into ManifoldSample
    point_cloud = torch.tensor(X_base, dtype=torch.float32)
    distance_tensor = torch.tensor(dist_matrix_norm, dtype=torch.float32)
    sample = ManifoldSample(point_cloud, distance_matrix=distance_tensor)

    # --- Step 4: Define theoretical K-function for 2D flat manifold
    theoretical_func = lambda r: np.pi * ((r * np.sqrt(0.5)) ** 2)

    # --- Step 5: Compute manifold score
    disagg_scores, agg_score = compute_manifold_score_on_torus(
        sample,
        theoretical_func,
        a=a,
        b=b,
        step=0.01,
        visualize=visualize,
    )

    return agg_score


# ========================================================
# =============== Experimental Runners ===================
# ========================================================

def run_test_flat_torus():
    num_points = 5000
    sample_type = "regular"
    visualize = True

    agg_score = exp_flat_torus(num_points, sample_type, visualize=visualize)
    print(f"\nAggregated Manifold Score for {num_points} points ({sample_type}): {agg_score:.4f}")

    plt.show()


def run_exp_flat_torus():
    num_points_list = [100, 500, 1000, 2000]
    sample_types = ["regular", "unif", "cross", "noisycross", "point"]
    num_repeats = 10

    results = []

    for n in num_points_list:
        for stype in sample_types:
            scores = []
            for _ in range(num_repeats):
                agg_score = exp_flat_torus(n, stype, a=0.0, b=0.7, visualize=False)
                scores.append(agg_score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results.append({
                "Num Points": n,
                "Sample Type": stype,
                "Mean Score": mean_score,
                "Std Dev": std_score
            })

    # --- Print formatted results table
    header = f"{'Num Points':>10} | {'Sample Type':>12} | {'Mean Score':>15} | {'Std Dev':>10}"
    print("\nManifold Score on Flat Torus Samples (aggregated score mean ± 1 std):\n")
    print(header)
    print("-" * len(header))

    for res in results:
        print(f"{res['Num Points']:>10} | {res['Sample Type']:>12} | {res['Mean Score']:>15.6f} | {res['Std Dev']:>10.6f}")


def main():
    # run_test_flat_torus()
    run_exp_flat_torus()
    plt.show()


if __name__ == "__main__":
    main()

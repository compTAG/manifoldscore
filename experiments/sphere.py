import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import Isomap
from scipy.special import gamma

from utils import compute_manifold_score

from shapes.sphere import generate_sphere_sample


def exp_2_sphere(num_points, sample_type, curvature_correction_func, a=0.0, b=0.3, visualize=False):
    sphere = generate_sphere_sample(n=num_points, d=3, sample_type=sample_type)
    theoretical_func = lambda r: np.pi * (r**2)  # K(r) = πr² for unit sphere

    disagg_scores, agg_score = compute_manifold_score(
        sphere,
        theoretical_func,
        disagg_correction=curvature_correction_func,
        agg_correction=curvature_correction_func,
        a=a,
        b=b,
        step=0.01,
        device='cpu',
        visualize=visualize
    )

    return agg_score

def run_test_2_sphere():
    num_points = 5000
    sample_type = "uniform"

    visualize = False

    a = 0.0
    b = 0.3

    EC = 2  # Euler characteristic for sphere
    curvature_correction_func = lambda r: (1.0 - (np.pi * EC / 24)) ** -1
    # curvature_correction_func = lambda r: 1.0

    agg_score_corrected = exp_2_sphere(num_points, sample_type, curvature_correction_func, visualize=visualize, a=a, b=b)
    agg_score_no_correction = exp_2_sphere(num_points, sample_type, lambda r: 1.0, visualize=visualize, a=a, b=b)

    print(f"Aggregated Manifold Score for {num_points} points ({sample_type} (with correction)): {agg_score_corrected:.4f}")
    print(f"Aggregated Manifold Score for {num_points} points ({sample_type} (no correction)): {agg_score_no_correction:.4f}")

def run_exp_2_sphere():
    num_points_list = [100, 500, 1000, 2000]
    sample_types = ["regular", "regular_jitter","uniform", "poles", "point"]

    sphere_radius = 1.0
    EC = 2  # Euler characteristic for sphere
    curvature_corrections = {
        # "with_correction": lambda r: 1.0 / (1.0 - (r**2) / (12.0 * sphere_radius**2)),
        "with_correction": lambda r: (1.0 - (np.pi * EC / 24)) ** -1,
        "no_correction": lambda r: 1.0
    }

    results = []

    num_repeats = 10

    for n in num_points_list:
        for stype in sample_types:
            for corr_name, corr_func in curvature_corrections.items():
                agg_scores = []
                for _ in range(num_repeats):
                    agg_score = exp_2_sphere(n, stype, corr_func, a=0.0, b=0.3, visualize=False)
                    agg_scores.append(agg_score)
                mean_score = np.mean(agg_scores)
                std_score = np.std(agg_scores)
                results.append({
                    "Num Points": n,
                    "Sample Type": stype,
                    "Curvature": corr_name,
                    "Mean Score": mean_score,
                    "Std Dev": std_score
                })

    # Print results as a nicely formatted table
    header = f"{'Num Points':>10} | {'Sample Type':>12} | {'Curvature':>15} | {'Mean Score':>15} | {'Std Dev':>10}"
    print("\nManifold Score on Different Samples of S^2 (aggregated score mean ± 1 std):\n")
    print(header)
    print("-" * len(header))

    for res in results:
        print(f"{res['Num Points']:>10} | {res['Sample Type']:>12} | {res['Curvature']:>15} | {res['Mean Score']:>15.6f} | {res['Std Dev']:>10.6f}")

def exp_n_sphere(num_points, d, sample_type, curvature_correction_func, a=0.0, b=0.3, visualize=False):
    sphere = generate_sphere_sample(n=num_points, d=d, sample_type=sample_type)

    # replace sphere distance matrix with one from the neighborhood graph (using Isomap from sklearn)
    isomap = Isomap(n_neighbors=10, n_components=d, metric='euclidean')
    isomap.fit(sphere.point_cloud.numpy())
    sphere.distance_matrix = torch.tensor(isomap.dist_matrix_ / isomap.dist_matrix_.max(), dtype=torch.float32)

    theoretical_func = lambda r: (np.pi ** (d / 2)) * (r**d) / gamma(d / 2 + 1)  # K(r) for unit sphere in R^d

    disagg_scores, agg_score = compute_manifold_score(
        sphere,
        theoretical_func,
        disagg_correction=curvature_correction_func,
        agg_correction=curvature_correction_func,
        a=a,
        b=b,
        step=0.01,
        device='cpu',
        visualize=visualize
    )

    return agg_score

def run_test_n_sphere():
    num_points = 1000
    sample_type = "uniform"
    d = 101  # Dimension of R^d, so sphere is S^(d-1)

    visualize = True

    a = 0.0
    b = 0.3

    LE = d - 1  # Euler characteristic for sphere
    curvature_correction_func = lambda r: (1.0 - (((r**2) * LE * (d - 2)) / (12 * (d - 1) * (d + 1)))) ** -1
    print(curvature_correction_func(0.5))

    agg_score_corrected = exp_n_sphere(num_points, d, sample_type, curvature_correction_func, visualize=visualize, a=a, b=b)
    agg_score_no_correction = exp_n_sphere(num_points, d, sample_type, lambda r: 1.0, visualize=visualize, a=a, b=b)

    print(f"Aggregated Manifold Score for {num_points} points ({sample_type} in R^{d} (with correction)): {agg_score_corrected:.8f}")
    print(f"Aggregated Manifold Score for {num_points} points ({sample_type} in R^{d} (no correction)): {agg_score_no_correction:.8f}")

def main():
    # run_exp_2_sphere()
    # run_test_2_sphere()

    run_test_n_sphere()

    plt.show()

if __name__ == "__main__":
    main()
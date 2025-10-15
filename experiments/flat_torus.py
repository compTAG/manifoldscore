import numpy as np
import matplotlib.pyplot as plt

from utils import compute_manifold_score

from shapes.flat_torus import (
    generate_flat_torus_sample,
)

# ========================================================
# =============== Flat Torus Experiment ==================
# ========================================================

def exp_flat_torus(num_points, sample_type, a=0.0, b=0.7, visualize=False):
    """
    Run a single manifold score experiment on the flat torus.

    Parameters
    ----------
    num_points : int
        Number of sampled points.
    sample_type : str
        One of {'regular', 'regular_jitter', 'unif', 'cross', 'noisycross', 'point'}.
    visualize : bool
        Whether to visualize the K-function comparison.

    Returns
    -------
    float
        Aggregated manifold score.
    """
    sample = generate_flat_torus_sample(num_points, sample_type)
    theoretical_func = lambda r: np.pi * ((r * np.sqrt(0.5)) ** 2)
    _, agg_score = compute_manifold_score(sample, theoretical_func, a=a, b=b, visualize=visualize, disagg_correction=lambda r: 1.0, agg_correction=lambda r: 1.0, step=0.01, device='cpu')
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
    sample_types = ["regular", "regular_jitter", "unif", "cross", "noisycross", "point"]
    num_repeats = 10

    results = []

    for n in num_points_list:
        for stype in sample_types:
            scores = []
            for _ in range(num_repeats):
                agg_score = exp_flat_torus(n, stype, visualize=False)
                scores.append(agg_score)

            results.append({
                "Num Points": n,
                "Sample Type": stype,
                "Mean Score": np.mean(scores),
                "Std Dev": np.std(scores)
            })

    header = f"{'Num Points':>10} | {'Sample Type':>15} | {'Mean Score':>15} | {'Std Dev':>10}"
    print("\nManifold Score on Flat Torus Samples (aggregated score mean ± 1 std):\n")
    print(header)
    print("-" * len(header))

    for res in results:
        print(f"{res['Num Points']:>10} | {res['Sample Type']:>15} | {res['Mean Score']:>15.6f} | {res['Std Dev']:>10.6f}")


def main():
    # run_test_flat_torus()
    run_exp_flat_torus()
    plt.show()


if __name__ == "__main__":
    main()

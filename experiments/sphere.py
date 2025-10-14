import numpy as np
import matplotlib.pyplot as plt

from manifoldscore.utilities import ManifoldSample
from manifoldscore.score import ManifoldScore

from shapes.sphere import generate_sphere_sample

def compute_manifold_score(sample: ManifoldSample, theoretical_func, disagg_correction, agg_correction, a=0.0, b=0.3, step=0.01, device='cpu', visualize=False):
    """
    Compute the manifold score for a given ManifoldSample against a theoretical function.

    Parameters:
    - sample: ManifoldSample instance containing the point cloud and distance matrix.
    - theoretical_func: function mapping radius to theoretical expected value.
    - disagg_correction: function for disaggregated correction based on radius.
    - agg_correction: function for aggregated correction based on radius.
    - a, b: range of radii to evaluate (from a to b).
    - step: step size for discretizing radii.
    - device: 'cpu', 'cuda', or 'mps' to indicate where tensors should reside.

    Returns:
    - disaggregated_scores: tensor of shape (N,) with score per point.
    - aggregated_score: scalar overall score.
    """
    scorer = ManifoldScore(sample, a=a, b=b, step=step, device=device)

    disagg_scores, agg_score = scorer.compute_scores(theoretical_func, aggregated_correction=agg_correction, disaggregated_correction=disagg_correction)

    if visualize:
        plot_kf_vs_radius(scorer, theoretical_func, title="K-function vs Radius")

    return disagg_scores, agg_score

def plot_kf_vs_radius(score_obj, theoretical_func=None, title="Neighborhood Function (K vs r)"):
    """
    Plot the empirical neighborhood function kf_vals against radius values.

    Parameters
    ----------
    score_obj : ManifoldScore
        A ManifoldScore instance after calling compute_scores().
    theoretical_func : callable, optional
        A function mapping radius -> theoretical expected K(r). 
        Example: lambda r: np.pi * (r**2)
    title : str
        Plot title.
    """
    # --- Ensure scores have been computed
    if score_obj.kf_vals is None:
        raise ValueError("kf_vals not found. Call `compute_scores()` on the ManifoldScore object first.")
    
    # --- Move tensors to CPU and convert to numpy
    radii = score_obj.radius_values.detach().cpu().numpy()
    kf_mean = score_obj.kf_vals.mean(dim=0).detach().cpu().numpy()
    kf_std = score_obj.kf_vals.std(dim=0).detach().cpu().numpy()

    # --- Create plot
    plt.figure(figsize=(7, 5))
    plt.plot(radii, kf_mean, label="Empirical K(r)", color="blue", linewidth=2)
    plt.fill_between(radii, kf_mean - kf_std, kf_mean + kf_std, color="blue", alpha=0.2, label="±1 std")

    # --- Optionally overlay theoretical curve
    if theoretical_func is not None:
        theor = np.array([theoretical_func(r) for r in radii])
        plt.plot(radii, theor, "--", color="red", linewidth=2, label="Theoretical K(r)")

    plt.title(title)
    plt.xlabel("Radius (r)")
    plt.ylabel("K(r)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()


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
    sample_types = ["regular", "uniform", "poles", "point"]

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


def main():
    run_exp_2_sphere()
    # run_test_2_sphere()
    plt.show()

if __name__ == "__main__":
    main()
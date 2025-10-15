import numpy as np
import matplotlib.pyplot as plt

from manifoldscore.score import ManifoldScore
from manifoldscore.utilities import ManifoldSample

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
import numpy as np
import torch
import matplotlib.pyplot as plt
from manifoldscore.utilities import ManifoldSample
from manifoldscore.score import ManifoldScore

np.random.seed(1234)

# --------------------------------------------------------
# ---  Geometry & data generation utilities
# --------------------------------------------------------

def generate_unit_square(n):
    return np.random.uniform(0, 1, (n, 2))

def generate_cross_stratification(n):
    rs = np.random.uniform(0, 1, n)
    half = n // 2
    top = np.column_stack((np.full(half, 0.5), rs[:half]))
    side = np.column_stack((rs[half:], np.full(n - half, 0.5)))
    return np.vstack((top, side))

def generate_cross_with_noise(n):
    noise = generate_unit_square(n // 10)
    cross = generate_cross_stratification(9 * (n // 10))
    return np.concatenate((noise, cross), axis=0)

def tesselate(X: np.ndarray) -> np.ndarray:
    """Tile points across 8 neighboring cells to simulate a flat torus."""
    n = X.shape[0]
    translations = {
        "w": (-1, 0), "nw": (-1, 1), "n": (0, 1), "ne": (1, 1),
        "e": (1, 0), "se": (1, -1), "s": (0, -1), "sw": (-1, -1)
    }

    for v in translations.values():
        X = np.vstack([X, X[:n] + v])

    return X

import matplotlib.pyplot as plt

def plot_tesselated(X, n, title="Tesselated Flat Torus"):
    """
    Visualize the tesselated manifold (flat torus) using color-coded tiles.
    
    Parameters:
    - X: NumPy array of shape (9n, 2)
    - n: number of original base points
    - title: optional title for the plot
    """
    colors = ["red", "blue", "green", "black", "cyan", "orange", "magenta", "gray", "yellow"]
    plt.figure(figsize=(6, 6))
    
    for i in range(9):
        start, end = n * i, n * (i + 1)
        plt.scatter(X[start:end, 0], X[start:end, 1], color=colors[i % len(colors)], s=10, alpha=0.6)
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()


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



# --------------------------------------------------------
# ---  Main scoring logic
# --------------------------------------------------------

def compute_manifold_score_on_flat_torus(n=100, type="unif", device='cpu'):
    """
    Generate a sampled manifold (flat torus) and compute its manifold score.
    """
    # --- Step 1: sample points
    if type == "unif":
        X = tesselate(generate_unit_square(n))
    elif type == "cross":
        X = tesselate(generate_cross_stratification(n))
    elif type == "noisycross":
        X = tesselate(generate_cross_with_noise(n))
    elif type == "point":
        # n copies of the same point
        X = tesselate(np.array([[0.5, 0.5]] * n))
    else:
        raise ValueError(f"Unknown type '{type}'")
    

    plot_tesselated(X, n, title=f"Tesselated Flat Torus - {type}")

    # --- Step 2: convert to tensor and create ManifoldSample
    point_cloud = torch.tensor(X, dtype=torch.float32)
    manifold = ManifoldSample(point_cloud, device=device)

    print("Max distance in distance matrix:", manifold.distance_matrix.max().item())
    print("Distance matrix", manifold.distance_matrix)

    # --- Step 3: define theoretical K-function for 2D uniform manifold
    # For a flat torus in 2D, K(r) ~ πr²
    theoretical_func = lambda r: np.pi * (r ** 2)

    # --- Step 4: compute score
    scorer = ManifoldScore(manifold, a=0.0, b=0.05, step=0.01, device=device)
    disagg_scores, agg_score = scorer.compute_scores(theoretical_func)

    print(disagg_scores.shape)
    print("median disaggregated score:", np.median(disagg_scores.numpy()))
    print("min disaggregated score:", np.min(disagg_scores.numpy()))
    print("max disaggregated score:", np.max(disagg_scores.numpy()))

    plot_kf_vs_radius(scorer, theoretical_func)

    return agg_score


# --------------------------------------------------------
# ---  Example: run several trials
# --------------------------------------------------------

def main():
    sample_size = 10
    num_trials = 1

    scores = np.zeros(num_trials)
    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}")
        scores[i] = compute_manifold_score_on_flat_torus(sample_size, type="point")

    print("\n===== RESULTS =====")
    print(f"num_trials: {num_trials}")
    print(f"sample_size: {sample_size}")
    print(f"avg: {np.mean(scores):.4f}")
    print(f"std: {np.std(scores):.4f}")

    plt.show()


if __name__ == "__main__":
    main()

# --------------------------------------------------------
# NOTES
# 
# I think the scorer is not properly using the tesselated function.
# The distance matrix is the size of the tesselated points, not just the base points.


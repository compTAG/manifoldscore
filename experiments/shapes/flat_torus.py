import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

# --------------------------------------------------------
# --- Geometry & data generation utilities (Unchanged)
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

    # Only tile the original points once (since X is overwritten in the loop, use X_original)
    X_original = X[:n]
    X_tesselated = X_original.copy()
    
    for v in translations.values():
        X_tesselated = np.vstack([X_tesselated, X_original + v])

    return X_tesselated

def toroidal_distance_matrix(X):
    """
    Compute pairwise toroidal distances for points in [0, 1]^2.
    Works without explicit tessellation, avoiding duplicate borders.
    """
    # Compute pairwise absolute differences
    delta = np.abs(X[:, None, :] - X[None, :, :])
    # Apply periodic boundary (minimum distance around the torus)
    delta = np.minimum(delta, 1 - delta)
    # Euclidean norm
    dist_matrix = np.sqrt((delta ** 2).sum(axis=2))
    return dist_matrix


def plot_tesselated(X, n, title="Tesselated Flat Torus"):
    """
    Visualize the tesselated manifold (flat torus) using color-coded tiles.
    """
    colors = ["red", "blue", "green", "black", "cyan", "orange", "magenta", "gray", "yellow"]
    plt.figure(figsize=(6, 6))
    
    for i in range(9):
        start, end = n * i, n * (i + 1)
        # Ensure we don't go out of bounds if n is small and the last block is incomplete
        if start < X.shape[0]:
            plt.scatter(X[start:min(end, X.shape[0]), 0], X[start:min(end, X.shape[0]), 1], 
                        color=colors[i % len(colors)], s=10, alpha=0.6)
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()


def plot_kf_vs_radius(score_obj, theoretical_func=None, title="Neighborhood Function (K vs r)"):
    # (Function implementation remains unchanged and works with ManifoldScore object)
    if score_obj.kf_vals is None:
        raise ValueError("kf_vals not found. Call `compute_scores()` on the ManifoldScore object first.")
    
    radii = score_obj.radius_values.detach().cpu().numpy()
    kf_mean = score_obj.kf_vals.mean(dim=0).detach().cpu().numpy()
    kf_std = score_obj.kf_vals.std(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(radii, kf_mean, label="Empirical K(r)", color="blue", linewidth=2)
    plt.fill_between(radii, kf_mean - kf_std, kf_mean + kf_std, color="blue", alpha=0.2, label="±1 std")

    if theoretical_func is not None:
        theor = np.array([theoretical_func(r) for r in radii])
        plt.plot(radii, theor, "--", color="red", linewidth=2, label="Theoretical K(r)")

    plt.title(title)
    plt.xlabel("Radius (r)")
    plt.ylabel("K(r)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()


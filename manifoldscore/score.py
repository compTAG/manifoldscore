import torch
from typing import Callable

# Import ManifoldSample class from the local utilities module
from .utilities import ManifoldSample

class ManifoldScore:
    """
    Class to compute scores comparing a sampled manifold's neighborhood structure 
    to a theoretical expectation over a range of radii.
    """
    def __init__(self, manifold_sample: ManifoldSample, a: float = 0.0, b: float = 0.3, step: float = 0.01, device='cpu'):
        """
        Initialize the ManifoldScore object.

        Parameters:
        - manifold_sample: instance of ManifoldSample containing the points and their distances.
        - a, b: range of radii to evaluate (from a to b).
        - step: step size for discretizing radii.
        - device: 'cpu', 'cuda', or 'mps' to indicate where tensors should reside.
        """
        self.device = device
        self.manifold_sample = manifold_sample
        self.a = a
        self.b = b
        self.step = step

        # Precompute discrete radius values as a tensor on the specified device
        self.radius_values = torch.arange(a, b, step, device=self.device)

        # Placeholders for computed results
        self.kf_vals = None               # Neighborhood counts normalized by N
        self.disaggregated_scores = None  # Score per point
        self.aggregated_score = None      # Overall score

    def compute_scores(self, 
                   theoretical_func: Callable[[float], float], 
                   aggregated_correction: Callable[[float], float] = lambda r: 1, 
                   disaggregated_correction: Callable[[float], float] = lambda r: 1):
        # Ensure radius values are on CPU for numpy operations
        radius_array = self.radius_values if self.device == 'cpu' else self.radius_values.cpu()
        radius_array = radius_array.numpy()

        # Evaluate the theoretical function and correction functions for each radius
        values = [(theoretical_func(r), aggregated_correction(r), disaggregated_correction(r)) for r in radius_array]
        theoretical, agg_correction, disagg_correction = zip(*values)

        # Convert results back to tensors on the correct device
        k_theoretical = torch.tensor(theoretical, device=self.device).view(1, -1)                 # Shape (1, R)
        agg_correction = torch.tensor(agg_correction, device=self.device).view(1, -1)           # Shape (1, R)
        disagg_correction = torch.tensor(disagg_correction, device=self.device).view(1, -1)     # Shape (1, R)

        # Load the pairwise distance matrix from the manifold sample
        distance_matrix = self.manifold_sample.distance_matrix.to(self.device)

        # Expand dimensions for broadcasting: (N, N, 1)
        distances_expanded = distance_matrix.unsqueeze(-1)  
        # Expand radius tensor for broadcasting: (1, 1, R)
        radius_expanded = self.radius_values.view(1, 1, -1)  

        # Count neighbors within each radius
        # Resulting shape: (N, R)
        within_radius = (distances_expanded <= radius_expanded).sum(dim=1)

        # Subtract 1 for self (distance zero), clamp to zero to avoid negatives
        within_radius = (within_radius - 1).clamp(min=0)

        # Normalize counts by total number of *other* points (N-1)
        kf_vals = within_radius.float() / float(max(1, self.manifold_sample.N - 1))

        self.kf_vals = kf_vals  # shape (N, R)

        # Compute disaggregated scores (per point)
        diff = (disagg_correction * kf_vals) - k_theoretical  # shape (N, R) after broadcasting
        # L2 norm across radii for each point
        norms = torch.norm(diff, dim=1)  # shape (N,)

        # Normalization factor: max possible L2 error across the radii.
        # Here we use the norm of a vector of ones (sqrt(R))
        R = k_theoretical.numel()
        normalization_factor = torch.sqrt(torch.tensor(float(R), device=self.device))
        
        # alternative normalization: ||k_theoretical||_2
        # normalization_factor = k_theoretical.numel() * torch.norm(k_theoretical, p=2)

        # Convert norms into similarity score in [0,1], clamp to [0,1]
        disagg_scores = 1.0 - norms
        disagg_scores = torch.clamp(disagg_scores, 0.0, 1.0)
        self.disaggregated_scores = disagg_scores

        # Compute aggregated score
        k_estimate_agg = agg_correction * kf_vals.mean(dim=0, keepdim=True)  # shape (1, R)
        agg_norm = torch.norm(k_estimate_agg - k_theoretical)  # scalar

        agg_score = 1.0 - (agg_norm / normalization_factor).item()
        agg_score = float(max(0.0, min(1.0, agg_score)))
        self.aggregated_score = agg_score

        return self.disaggregated_scores, self.aggregated_score

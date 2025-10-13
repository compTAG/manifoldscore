import torch
from typing import Callable

from .utilities import ManifoldSample

class ManifoldScore:
    def __init__(self, manifold_sample: ManifoldSample, a: float = 0.0, b: float = 0.3, step: float = 0.01, device='cpu'):
        self.device = device

        self.manifold_sample = manifold_sample
        self.a = a
        self.b = b
        self.step = step

        self.radius_values = torch.arange(a, b, step, device=self.device)
        self.kf_vals = None
        self.disaggregated_scores = None
        self.aggregated_score = None

    def compute_scores(self, theoretical_func: Callable[[float], float], aggregated_correction: Callable[[float], float] =  lambda r: 1, disaggregated_correction: Callable[[float], float] = lambda r: 1):
        radius_array = self.radius_values if self.device == 'cpu' else self.radius_values.cpu()
        radius_array = radius_array.numpy()

        values = [(theoretical_func(r), aggregated_correction(r), disaggregated_correction(r)) for r in radius_array]
        theoretical, agg_correction, disagg_correction = zip(*values)
        k_theoretical = torch.tensor(theoretical, device=self.device)
        agg_correction = torch.tensor(agg_correction, device=self.device).view(1, -1)
        disagg_correction = torch.tensor(disagg_correction, device=self.device).view(1, -1)

        # Load the distance matrix
        distance_matrix = self.manifold_sample.distance_matrix.to(self.device)

        # Broadcast radius comparison
        # Shape: (N, N, R) where R = number of radius steps
        distances_expanded = distance_matrix.unsqueeze(-1)  # (N, N, 1)
        radius_expanded = self.radius_values.view(1, 1, -1)  # (1, 1, R)

        # Count neighbors within radius for each point
        within_radius = (distances_expanded <= radius_expanded).sum(dim=1) - 1  # exclude self

        # Normalize
        kf_vals = within_radius.float() / self.manifold_sample.N
        self.kf_vals = kf_vals

        # Disaggregated scores (vectorized)
        diff = (disagg_correction * kf_vals) - k_theoretical  # (N, R)
        norms = torch.norm(diff, dim=1)
        self.disaggregated_scores = (1 - norms / self.manifold_sample.N)

        # Aggregated score
        k_estimate_agg = agg_correction * kf_vals.mean(dim=0)
        agg_norm = torch.norm(k_estimate_agg - k_theoretical)
        self.aggregated_score = 1 - (agg_norm / self.manifold_sample.N).item()

        return self.disaggregated_scores, self.aggregated_score
import numpy as np
import torch
import pytest
from manifoldscore.score import ManifoldScore
from manifoldscore.utilities import ManifoldSample


def test_score_simple():
    """Test basic ManifoldScore computation with a small point cloud."""
    
    # Small 2D point cloud
    point_cloud = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [1.0, 1.0],
    ], dtype=torch.float32)

    # Create ManifoldSample
    sample = ManifoldSample(point_cloud)

    # Create ManifoldScore
    score = ManifoldScore(sample, a=0.0, b=0.3, step=0.01)

    # Define theoretical function and correction
    theoretical = lambda r: np.pi * r ** 2
    correction = lambda r: r

    # Compute scores
    disaggregated_scores, aggregated_score = score.compute_scores(
        theoretical_func=theoretical,
        aggregated_correction=correction,
        disaggregated_correction=correction
    )

    # Assertions
    assert isinstance(disaggregated_scores, torch.Tensor)
    assert disaggregated_scores.shape[0] == sample.N
    assert isinstance(aggregated_score, float)
    assert 0.0 <= aggregated_score <= 1.0

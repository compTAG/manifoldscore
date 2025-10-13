import pytest
import torch
from manifoldscore.utilities import ManifoldSample


@pytest.fixture
def random_point_cloud():
    # Create a 5x3 tensor of random points
    return torch.rand(5, 3)


def test_initialization_with_tensor(random_point_cloud):
    """Ensure ManifoldSample initializes correctly with a valid tensor."""
    sample = ManifoldSample(random_point_cloud)
    assert isinstance(sample.point_cloud, torch.Tensor)
    assert sample.N == random_point_cloud.shape[0]
    assert sample.distance_matrix.shape == (sample.N, sample.N)


def test_initialization_with_non_tensor():
    """Should raise TypeError if non-tensor is passed."""
    with pytest.raises(TypeError):
        ManifoldSample([[0.1, 0.2, 0.3]])


def test_distance_matrix_is_symmetric(random_point_cloud):
    """Distance matrix should be symmetric."""
    sample = ManifoldSample(random_point_cloud)
    assert torch.allclose(sample.distance_matrix, sample.distance_matrix.T, atol=1e-6)


def test_distance_matrix_self_zero(random_point_cloud):
    """Distance from a point to itself should be zero."""
    sample = ManifoldSample(random_point_cloud)
    diagonal = torch.diag(sample.distance_matrix)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)


def test_distance_matrix_normalized(random_point_cloud):
    """Distance matrix should be normalized to max = 1."""
    sample = ManifoldSample(random_point_cloud)
    max_val = sample.distance_matrix.max().item()
    assert pytest.approx(max_val, rel=1e-5) == 1.0


def test_custom_distance_matrix_override(random_point_cloud):
    """If distance_matrix is provided, it should be used directly."""
    custom_matrix = torch.ones((5, 5))
    sample = ManifoldSample(random_point_cloud, distance_matrix=custom_matrix)
    assert torch.allclose(sample.distance_matrix, custom_matrix / custom_matrix.max())


def test_compute_distance_matrix_function(random_point_cloud):
    """compute_distance_matrix() should return valid normalized distances."""
    sample = ManifoldSample(random_point_cloud)
    computed = sample.compute_distance_matrix()
    assert computed.shape == (sample.N, sample.N)
    assert torch.allclose(computed, computed.T, atol=1e-6)
    assert torch.all(computed >= 0)
    assert pytest.approx(computed.max().item(), rel=1e-5) == 1.0


def test_device_movement(random_point_cloud):
    """Ensure point cloud and distance matrix are moved to the correct device."""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    sample = ManifoldSample(random_point_cloud, device=device)
    assert sample.point_cloud.device.type == device
    assert sample.distance_matrix.device.type == device

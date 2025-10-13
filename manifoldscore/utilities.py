import torch

class ManifoldSample:
    """
    Represents a sampled manifold via a point cloud and its pairwise distance matrix.
    Provides utilities to compute distances and ensure normalization.
    """
    def __init__(self, point_cloud: torch.Tensor, distance_matrix: torch.Tensor = None, device='cpu'):
        """
        Initialize a ManifoldSample instance.

        Parameters:
        - point_cloud: tensor of shape (N, D) representing N points in D-dimensional space.
        - distance_matrix: optional precomputed NxN distance matrix.
        - device: 'cpu' or 'cuda', specifies where tensors will reside.
        """
        # Ensure input is a torch tensor
        if not isinstance(point_cloud, torch.Tensor):
            raise TypeError("point_cloud must be a torch.Tensor")

        # Move point cloud to specified device
        self.point_cloud = point_cloud.to(device)
        # Store number of points
        self.N = self.point_cloud.shape[0]

        # If distance matrix not provided, compute it
        self.distance_matrix = self.compute_distance_matrix() if distance_matrix is None else distance_matrix

        # Redundant type check (can be removed, already checked above)
        if type(self.point_cloud) is not torch.Tensor:
            raise TypeError("point_cloud must be a torch.Tensor")
        
        # Normalize distance matrix to have max value 1
        if self.distance_matrix.max() != 1:
            print("Distance matrix is not normalized. Normalizing...")
            self.distance_matrix = self.distance_matrix / self.distance_matrix.max()


    def compute_distance_matrix(self):
        """
        Compute the pairwise Euclidean distance matrix for the point cloud.

        Returns:
        - dist_matrix: NxN tensor of distances, normalized so max distance is 1.
        """
        x = self.point_cloud  # Shape: (N, D)
        
        # Compute squared norms for each point: (x_i^2 summed over dimensions)
        # Shape: (N, 1)
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        
        # Use the formula ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
        dist_matrix = x_norm + x_norm.T - 2.0 * (x @ x.T)
        
        # Clamp negative values due to numerical errors to 0
        dist_matrix = torch.clamp(dist_matrix, min=0.0)
        
        # Take square root to get Euclidean distances
        dist_matrix = torch.sqrt(dist_matrix)

        # Normalize distance matrix so max distance is 1
        max_val = dist_matrix.max()
        if max_val > 0:
            dist_matrix = dist_matrix / max_val

        return dist_matrix

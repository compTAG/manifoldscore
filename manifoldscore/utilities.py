import torch

class ManifoldSample:
    def __init__(self, point_cloud: torch.Tensor, distance_matrix: torch.Tensor = None, device='cpu'):
        if not isinstance(point_cloud, torch.Tensor):
            raise TypeError("point_cloud must be a torch.Tensor")

        self.point_cloud = point_cloud.to(device)
        self.N = self.point_cloud.shape[0]
        self.distance_matrix = self.compute_distance_matrix() if distance_matrix is None else distance_matrix

        if type(self.point_cloud) is not torch.Tensor:
            raise TypeError("point_cloud must be a torch.Tensor")
        
        # ensure distance matrix is normalized
        if self.distance_matrix.max() != 1:
            print("Distance matrix is not normalized. Normalizing...")
            self.distance_matrix = self.distance_matrix / self.distance_matrix.max()


    def compute_distance_matrix(self):
        """
        Compute the pairwise distance matrix for the point cloud.
        """
        x = self.point_cloud
        # x: (N, D)
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # (N, 1)
        dist_matrix = x_norm + x_norm.T - 2.0 * (x @ x.T)
        dist_matrix = torch.clamp(dist_matrix, min=0.0)  # Numerical stability
        dist_matrix = torch.sqrt(dist_matrix)

        max_val = dist_matrix.max()
        if max_val > 0:
            dist_matrix = dist_matrix / max_val

        return dist_matrix

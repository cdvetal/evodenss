import torch
from torch import nn, Tensor
import torch.nn.functional as F

class BarlowTwinsLoss(nn.Module):

    def __init__(self, lamb: float):
        super(BarlowTwinsLoss, self).__init__()
        self.lamb = lamb

    def forward(self, z_a: Tensor, z_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # normalize repr. along the batch dimension
        z_a_norm, z_b_norm = self._normalize(z_a, z_b)
        batch_size: int = z_a.size(0)

        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c[c.isnan()] = 0.0
        valid_c = c[~c.isinf()]
        limit = 1e+30 if c.dtype == torch.float32 else 1e+4
        try:
            max_value = torch.max(valid_c)
        except RuntimeError:
            max_value = limit # type: ignore
        try:
            min_value = torch.min(valid_c)
        except RuntimeError:
            min_value = -limit # type: ignore
        c[c == float("Inf")] = max_value if max_value != 0.0 else limit
        c[c == float("-Inf")] = min_value if min_value != 0.0 else -limit
        c.div_(batch_size)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = self._off_diagonal(c).pow_(2).sum()
        loss: Tensor = invariance_loss + self.lamb * redundancy_reduction_loss
        return loss, invariance_loss, redundancy_reduction_loss


    def _normalize(self, z_a: Tensor, z_b: Tensor) -> tuple[Tensor, Tensor]:
        """Helper function to normalize tensors along the batch dimension."""
        combined = torch.stack([z_a, z_b], dim=0)  # Shape: 2 x N x D
        normalized = F.batch_norm(
            combined.flatten(0, 1),
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            training=True,
        ).view_as(combined)
        return normalized[0], normalized[1]


    def _off_diagonal(self, x: Tensor) -> Tensor:
        # return a flattened view of the off-diagonal elements of a square matrix
        n: int
        m: int
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

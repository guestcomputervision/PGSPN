
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK

@LOSS_BLOCK.register_module()
class ScaleInvarintLoss(nn.Module):
    def __init__(self,
                 lambda_gs_l1_v2: float = 1.0,
                 loss_type: str = 'l1',
                 detach_sparse: bool = False,
                 scale_min_threshold: float = 0.01,
                 scale_max_threshold: float = 25.0,
                 distance_weighted: bool = True,
                 weight_power: float = 1.0):
        super().__init__()
        self.lambda_gs_l1_v2 = lambda_gs_l1_v2
        self.loss_type = loss_type
        self.detach_sparse = detach_sparse
        self.scale_min_threshold = scale_min_threshold
        self.scale_max_threshold = scale_max_threshold
        self.distance_weighted = distance_weighted
        self.weight_power = weight_power

    def _compute_global_scale_and_bias(self, dense_img, sparse_depth, eps=1e-6):
        B, C, _, _ = dense_img.shape

        # Create mask for valid sparse depth points within [scale_min_threshold, scale_max_threshold]
        valid_min = sparse_depth > self.scale_min_threshold
        valid_max = sparse_depth <= self.scale_max_threshold
        sparse_mask = (valid_min & valid_max).float()  # (B, 1, H, W)

        # Check if we have any valid pixels
        valid_count = sparse_mask.sum(dim=[2, 3])  # (B, 1)
        has_valid_pixels = valid_count > 1  # Need at least 2 points for scale+bias

        # Flatten spatial dimensions for computation
        dense_flat = dense_img.view(B, C, -1)  # (B, C, H*W)
        sparse_flat = sparse_depth.view(B, 1, -1)  # (B, 1, H*W)
        mask_flat = sparse_mask.view(B, 1, -1)  # (B, 1, H*W)

        # Compute distance weights if enabled - closer depths get higher weights
        if self.distance_weighted:
            # Create inverse distance weights (closer = higher weight)
            # Use 1/depth^power weighting, but clamp to avoid extreme values
            depth_weights = torch.pow(1.0 / torch.clamp(sparse_flat, min=self.scale_min_threshold), self.weight_power)
            # Apply mask to weights
            depth_weights = depth_weights * mask_flat  # (B, 1, H*W)
        else:
            depth_weights = mask_flat  # (B, 1, H*W)

        # Solve weighted least squares for scale and bias: [s, b] = (A^T W A)^(-1) A^T W d
        # where A = [dense_values, ones], W is diagonal weight matrix, d is sparse_depth

        # Create design matrix A for each channel
        ones_flat = torch.ones_like(dense_flat)  # (B, C, H*W)

        # Apply weights to all terms
        weighted_dense = dense_flat * depth_weights  # (B, C, H*W)
        weighted_ones = ones_flat * depth_weights  # (B, C, H*W)
        weighted_sparse = sparse_flat * depth_weights  # (B, 1, H*W)

        # Compute A^T W A matrix elements for 2x2 system per channel
        # [sum(w*a^2)   sum(w*a)  ] [s]   [sum(w*a*d)]
        # [sum(w*a)    sum(w)    ] [b] = [sum(w*d)  ]

        sum_w_a2 = (weighted_dense * dense_flat).sum(dim=-1)  # (B, C)
        sum_w_a = (weighted_dense).sum(dim=-1)  # (B, C)
        sum_w = (weighted_ones).sum(dim=-1)  # (B, C)
        sum_w_ad = (weighted_dense * sparse_flat).sum(dim=-1)  # (B, C)
        sum_w_d = (weighted_sparse).sum(dim=-1)  # (B, 1) -> broadcast to (B, C)
        sum_w_d = sum_w_d.expand(-1, C)  # (B, C)

        # Compute determinant for 2x2 matrix inversion
        det = sum_w_a2 * sum_w - sum_w_a * sum_w_a + eps  # (B, C)

        # Solve for scale and bias using Cramer's rule
        scale = (sum_w * sum_w_ad - sum_w_a * sum_w_d) / det  # (B, C)
        bias = (sum_w_a2 * sum_w_d - sum_w_a * sum_w_ad) / det  # (B, C)

        return torch.clamp(scale, min=1e-6), bias, sparse_mask, has_valid_pixels

    def forward(self, x):
        """
        Forward pass for GS L1 v2 Loss with global scale and bias correction.

        Args:
            x[0]: depth_dependent_img - (B, C>=3, H, W) depth dependent image
            x[1]: sparse_depth - (B, 1, H, W) sparse depth map
            x[2]: valid_mask - (B, 1, H, W) valid mask for loss computation

        Returns:
            loss: Global scale and bias invariant consistency loss
        """
        depth_dependent_img = x[0]  # Single tensor
        sparse_depth = x[1]  # (B, 1, H, W)
        valid_mask = x[2]  # (B, 1, H, W)

        assert depth_dependent_img.dim() == 4, "depth_dependent_img should be 4D tensor"
        assert sparse_depth.dim() == 4, "sparse_depth should be 4D tensor"

        if self.detach_sparse:
            sparse_depth = sparse_depth.detach()

        # Take first 3 channels and apply log transform
        processed_img = -1.0 * torch.log(depth_dependent_img[:, :3, :, :] + 1e-8)  # (B, 3, H, W)

        # Compute global scale and bias factors using only valid depth range
        scale_factors, bias_factors, sparse_mask, _ = self._compute_global_scale_and_bias(processed_img, sparse_depth)  # (B, 3), (B, 3), (B, 1, H, W)

        # Skip loss computation if no valid sparse pixels exist
        total_valid_pixels = sparse_mask.sum()
        if total_valid_pixels == 0:
            return torch.tensor(0.0, device=sparse_depth.device, requires_grad=True)

        # Apply scale and bias correction
        scale_expanded = scale_factors.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        bias_expanded = bias_factors.unsqueeze(-1).unsqueeze(-1)    # (B, 3, 1, 1)
        corrected_img = scale_expanded * processed_img + bias_expanded  # (B, 3, H, W)

        # Compute residual only at valid sparse locations
        residual = (corrected_img - sparse_depth) * valid_mask  # (B, 3, H, W)

        # Compute loss only for valid pixels
        if self.loss_type == 'l1':
            loss = residual.abs().sum() / (total_valid_pixels * 3)
        else:  # l2
            loss = residual.pow(2).sum() / (total_valid_pixels * 3)

        return self.lambda_gs_l1_v2 * loss
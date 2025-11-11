import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings


class DepthScaleCorrector(nn.Module):
    def __init__(self,
                 max_depth: float = 20.0,
                 valid_threshold: float = 1e-6,
                 min_valid_points: int = 10,
                 enable_logging: bool = False):
        super(DepthScaleCorrector, self).__init__()

        self.max_depth = max_depth
        self.valid_threshold = valid_threshold
        self.min_valid_points = min_valid_points
        self.enable_logging = enable_logging


    def estimate_scale_bias_vectorized(self, non_scale_dense, sparse_depth):
        if non_scale_dense.dim() == 4:
            if non_scale_dense.size(1) == 1:
                non_scale_dense = non_scale_dense.squeeze(1)  # [B, H, W]
                multi_channel = False
            else:
                multi_channel = True
        else:
            multi_channel = False

        if sparse_depth.dim() == 4:
            sparse_depth = sparse_depth.squeeze(1)

        batch_size = non_scale_dense.size(0)
        device = non_scale_dense.device

        if multi_channel:
            num_channels = non_scale_dense.size(1)
            return self._estimate_multichannel_scale_bias(non_scale_dense, sparse_depth)
        else:
            scales, biases = self._estimate_single_channel_scale_bias(non_scale_dense, sparse_depth)
            return scales.unsqueeze(1), biases.unsqueeze(1)

    def _estimate_multichannel_scale_bias(self, non_scale_dense, sparse_depth):
        batch_size, num_channels = non_scale_dense.size(0), non_scale_dense.size(1)
        device = non_scale_dense.device

        valid_mask = (sparse_depth > self.valid_threshold) & \
                     (sparse_depth <= self.max_depth)

        valid_counts = valid_mask.sum(dim=[1, 2])  # [B]
        insufficient_mask = valid_counts < self.min_valid_points

        if insufficient_mask.any() and self.enable_logging:
            insufficient_batches = insufficient_mask.nonzero(as_tuple=True)[0]
            for b_idx in insufficient_batches:
                warnings.warn(f"Not enough valid points in batch {b_idx}: "
                            f"{valid_counts[b_idx]} < {self.min_valid_points}")

        scales = torch.ones(batch_size, num_channels, device=device)
        biases = torch.zeros(batch_size, num_channels, device=device)

        valid_batches = ~insufficient_mask
        if not valid_batches.any():
            return scales, biases
        
        valid_non_scale = non_scale_dense[valid_batches]  # [B_valid, 3, H, W]
        valid_sparse = sparse_depth[valid_batches]  # [B_valid, H, W]
        valid_mask_sub = valid_mask[valid_batches]  # [B_valid, H, W]

        computed_scales = []
        computed_biases = []

        for i, (non_scale_batch, sparse_batch, mask_batch) in enumerate(
            zip(valid_non_scale, valid_sparse, valid_mask_sub)):

            y = sparse_batch[mask_batch]  # [N_valid] - sparse depth values

            if y.numel() == 0:
                computed_scales.append(torch.ones(num_channels, device=device))
                computed_biases.append(torch.zeros(num_channels, device=device))
                continue

            batch_scales = []
            batch_biases = []

            for c in range(num_channels):
                x = non_scale_batch[c][mask_batch]  # [N_valid] - depth dependent values for channel c

                n = x.size(0)

                x_sum = x.sum()
                x_sq_sum = (x * x).sum()
                y_sum = y.sum()
                xy_sum = (x * y).sum()

                det = n * x_sq_sum - x_sum * x_sum

                if det.abs() < 1e-8:
                    if self.enable_logging:
                        warnings.warn(f"Near-singular matrix detected in batch {i}, channel {c}")
                    batch_scales.append(torch.tensor(1.0, device=device))
                    batch_biases.append(torch.tensor(0.0, device=device))
                else:
                    scale = (n * xy_sum - x_sum * y_sum) / det
                    bias = (x_sq_sum * y_sum - x_sum * xy_sum) / det
                    batch_scales.append(scale)
                    batch_biases.append(bias)

            computed_scales.append(torch.stack(batch_scales))
            computed_biases.append(torch.stack(batch_biases))

        if computed_scales:
            scales[valid_batches] = torch.stack(computed_scales)
            biases[valid_batches] = torch.stack(computed_biases)

        return scales, biases

    def _estimate_single_channel_scale_bias(self,
                                          non_scale_dense: torch.Tensor,
                                          sparse_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = non_scale_dense.size(0)
        device = non_scale_dense.device

        valid_mask = (sparse_depth > self.valid_threshold) & \
                     (sparse_depth <= self.max_depth)

        valid_counts = valid_mask.sum(dim=[1, 2])  # [B]
        insufficient_mask = valid_counts < self.min_valid_points

        if insufficient_mask.any() and self.enable_logging:
            insufficient_batches = insufficient_mask.nonzero(as_tuple=True)[0]
            for b_idx in insufficient_batches:
                warnings.warn(f"Not enough valid points in batch {b_idx}: "
                            f"{valid_counts[b_idx]} < {self.min_valid_points}")

        scales = torch.ones(batch_size, device=device)
        biases = torch.zeros(batch_size, device=device)

        valid_batches = ~insufficient_mask
        if not valid_batches.any():
            return scales, biases

        valid_non_scale = non_scale_dense[valid_batches]  # [B_valid, H, W]
        valid_sparse = sparse_depth[valid_batches]  # [B_valid, H, W]
        valid_mask_sub = valid_mask[valid_batches]  # [B_valid, H, W]

        computed_scales = []
        computed_biases = []

        for i, (non_scale_batch, sparse_batch, mask_batch) in enumerate(
            zip(valid_non_scale, valid_sparse, valid_mask_sub)):

            x = non_scale_batch[mask_batch]  # [N_valid]
            y = sparse_batch[mask_batch]     # [N_valid]

            if x.numel() == 0:
                computed_scales.append(torch.tensor(1.0, device=device))
                computed_biases.append(torch.tensor(0.0, device=device))
                continue

            n = x.size(0)

            x_sum = x.sum()
            x_sq_sum = (x * x).sum()
            y_sum = y.sum()
            xy_sum = (x * y).sum()

            det = n * x_sq_sum - x_sum * x_sum

            if det.abs() < 1e-8:
                if self.enable_logging:
                    warnings.warn(f"Near-singular matrix detected in batch {i}")
                computed_scales.append(torch.tensor(1.0, device=device))
                computed_biases.append(torch.tensor(0.0, device=device))
            else:
                scale = (n * xy_sum - x_sum * y_sum) / det
                bias = (x_sq_sum * y_sum - x_sum * xy_sum) / det
                computed_scales.append(scale)
                computed_biases.append(bias)

        if computed_scales:
            scales[valid_batches] = torch.stack(computed_scales)
            biases[valid_batches] = torch.stack(computed_biases)

        return scales, biases

    def estimate_scale_bias(self,
                           non_scale_dense: torch.Tensor,
                           sparse_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.estimate_scale_bias_vectorized(non_scale_dense, sparse_depth)

    def correct_depth(self,
                     non_scale_dense: torch.Tensor,
                     scales: torch.Tensor,
                     biases: torch.Tensor) -> torch.Tensor:

        original_shape = non_scale_dense.shape

        if non_scale_dense.dim() == 4:
            if non_scale_dense.size(1) == 1:
                # [B, 1, H, W] -> [B, H, W]
                non_scale_dense = non_scale_dense.squeeze(1)
                single_channel = True
            else:
                single_channel = False
        else:
            # [B, H, W]
            single_channel = True

        if single_channel:
            if scales.dim() == 2:
                scales = scales.squeeze(1)  # [B, 1] -> [B]
                biases = biases.squeeze(1)  # [B, 1] -> [B]

            scales = scales.view(-1, 1, 1)  # [B, 1, 1]
            biases = biases.view(-1, 1, 1)  # [B, 1, 1]

            corrected_depth = scales * non_scale_dense + biases

            if len(original_shape) == 4 and original_shape[1] == 1:
                corrected_depth = corrected_depth.unsqueeze(1)
        else:
            batch_size, num_channels, height, width = non_scale_dense.shape

            scales = scales.view(batch_size, num_channels, 1, 1)  # [B, 3, 1, 1]
            biases = biases.view(batch_size, num_channels, 1, 1)  # [B, 3, 1, 1]

            corrected_depth = scales * non_scale_dense + biases

        corrected_depth = torch.clamp(corrected_depth, min=0.0, max=self.max_depth)

        return corrected_depth

    def forward(self,
                non_scale_dense: torch.Tensor,
                sparse_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        scales, biases = self.estimate_scale_bias_vectorized(non_scale_dense, sparse_depth)

        corrected_depth = self.correct_depth(non_scale_dense, scales, biases)

        best_corrected, best_channel_indices = self._find_best_corrected_channel(
            non_scale_dense, sparse_depth, scales, biases
        )
        
        return corrected_depth, best_corrected, best_channel_indices

    def extract_best_channels(self,
                            multi_channel_tensor: torch.Tensor,
                            best_channel_indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = multi_channel_tensor.size()
        device = multi_channel_tensor.device

        extracted_channels = torch.zeros(batch_size, 1, height, width, device=device)

        for b in range(batch_size):
            channel_idx = best_channel_indices[b].item()
            extracted_channels[b, 0] = multi_channel_tensor[b, channel_idx]

        return extracted_channels

    def _find_best_corrected_channel(self,
                                    non_scale_dense: torch.Tensor,
                                    sparse_depth: torch.Tensor,
                                    scales: torch.Tensor,
                                    biases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = non_scale_dense.size()
        device = non_scale_dense.device

        if sparse_depth.dim() == 4:
            sparse_depth = sparse_depth.squeeze(1)  # [B, H, W]

        valid_mask = (sparse_depth > self.valid_threshold) & (sparse_depth <= self.max_depth)

        mse_per_channel = torch.full((batch_size, num_channels), float('inf'), device=device)

        for b in range(batch_size):
            mask = valid_mask[b]
            if not mask.any():
                continue

            sparse_valid = sparse_depth[b][mask]

            for c in range(num_channels):
                corrected = scales[b, c] * non_scale_dense[b, c] + biases[b, c]
                corrected_valid = corrected[mask]

                mse = ((corrected_valid - sparse_valid) ** 2).mean()
                mse_per_channel[b, c] = mse

        best_channel_indices = mse_per_channel.argmin(dim=1)  # [B]
        best_corrected = torch.zeros(batch_size, 1, height, width, device=device)

        for b in range(batch_size):
            best_c = best_channel_indices[b].item()
            corrected_channel = scales[b, best_c] * non_scale_dense[b, best_c] + biases[b, best_c]
            corrected_channel = torch.clamp(corrected_channel, min=0.0, max=self.max_depth)
            best_corrected[b, 0] = corrected_channel

        return best_corrected, best_channel_indices



if __name__ == "__main__":
    import time

    batch_size, height, width = 4, 480, 640  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Testing with batch size: {batch_size}")

    corrector = DepthScaleCorrector(enable_logging=True).to(device)

    num_iterations = 10
    total_time = 0

    for i in range(num_iterations):
        non_scale_dense = torch.abs(torch.randn(batch_size, 3, height, width, device=device)) * 5
        sparse_depth = torch.abs(torch.randn(batch_size, height, width, device=device)) * 10

        mask = torch.rand(batch_size, height, width, device=device) > 0.7
        sparse_depth = sparse_depth * mask.float()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        corrected_depth, best_corrected, best_channel_indices = corrector(non_scale_dense, sparse_depth)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        iteration_time = time.time() - start_time
        total_time += iteration_time

        if i == 0:
            print(f"Input dense depth shape: {non_scale_dense.shape}")
            print(f"Input sparse depth shape: {sparse_depth.shape}")
            print(f"Corrected depth shape: {corrected_depth.shape}")
            print(f"Valid sparse points per batch: {(sparse_depth > 0).sum(dim=[1,2]).cpu().numpy()}")

        print(f"Iteration {i+1}: {iteration_time:.4f}s")

    avg_time = total_time / num_iterations
    fps = batch_size / avg_time  # frames per second

    resolutions = [(240, 320), (480, 640), (720, 1280)]

    for h, w in resolutions:
        test_dense = torch.abs(torch.randn(2, h, w, device=device)) * 5
        test_sparse = torch.abs(torch.randn(2, h, w, device=device)) * 10
        test_mask = torch.rand(2, h, w, device=device) > 0.8
        test_sparse = test_sparse * test_mask.float()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        test_corrected = corrector(test_dense, test_sparse)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        test_time = time.time() - start_time

        print(f"Resolution {h}x{w}: {test_time:.4f}s ({2/test_time:.1f} FPS)")



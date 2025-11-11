
from .base_l1_loss import L1_loss, Depth_Estimation_L1_loss, Iter_Depth_Estimation_L1_loss
from .eh_ssim_loss import SSIMLoss
from .scale_invariant_loss import ScaleInvarintLoss

# Define L2_loss as a placeholder or alias
L2_loss = L1_loss  # Use L1_loss as a placeholder for L2_loss

__all__ = [
    'Depth_Estimation_L1_loss', 'Iter_Depth_Estimation_L1_loss',
    'L1_loss', 'L2_loss',
    'SSIMLoss',
    'ScaleInvarintLoss'
    ]
    
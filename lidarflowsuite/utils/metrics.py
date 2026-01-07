import torch

def compute_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
    """Computes End-Point Error (EPE)."""
    error = torch.norm(pred_flow - gt_flow, p=2, dim=-1)
    return error.mean()

def compute_accuracy(pred_flow: torch.Tensor, gt_flow: torch.Tensor, threshold: float = 0.05, relative_threshold: float = 0.05) -> torch.Tensor:
    """Computes accuracy metrics (Acc Relaxed/Strict type)."""
    error = torch.norm(pred_flow - gt_flow, p=2, dim=-1)
    gt_norm = torch.norm(gt_flow, p=2, dim=-1)
    
    mask = (error < threshold) | (error < relative_threshold * gt_norm)
    return mask.float().mean()

import torch
import torch.nn as nn

from lidarflowsuite.models.layers import find_knn, group_points

def chamfer_distance(pc1, pc2):
    """
    Simplified Chamfer Distance between two point clouds.
    pc1: [B, N, 3], pc2: [B, M, 3]
    """
    dist = torch.norm(pc1.unsqueeze(2) - pc2.unsqueeze(1), dim=-1, p=2)
    dist1 = torch.min(dist, dim=2)[0]
    dist2 = torch.min(dist, dim=1)[0]
    return dist1.mean() + dist2.mean()

def smoothness_loss(pc, flow, k=8):
    """
    Encourages nearby points to have similar flow vectors.
    """
    idx = find_knn(pc, k=k+1)[:, :, 1:] # [B, N, k]
    neighbor_flow = group_points(flow, idx) # [B, N, k, 3]
    diff = flow.unsqueeze(2) - neighbor_flow
    return torch.norm(diff, p=2, dim=-1).mean()

def cycle_consistency_loss(pc1, pc2, flow12, model, device):
    """
    Ensures that forward flow + backward flow returns to origin.
    """
    pc1_warped = pc1 + flow12
    # Estimate backward flow from pc1_warped to pc1
    # Note: In a real training, we'd pass this through the model again
    # This requires the model to be passed or accessible.
    flow21 = model(pc1_warped, pc1)
    cycle_error = torch.norm(flow12 + flow21, p=2, dim=-1)
    return cycle_error.mean()

class SceneFlowLoss(nn.Module):
    def __init__(self, w_chamfer=1.0, w_smooth=0.5, w_cycle=0.1):
        super(SceneFlowLoss, self).__init__()
        self.w_chamfer = w_chamfer
        self.w_smooth = w_smooth
        self.w_cycle = w_cycle

    def forward(self, pc1, pc2, pred_flow, model=None):
        pc1_warped = pc1 + pred_flow
        
        l_chamfer = chamfer_distance(pc1_warped, pc2)
        l_smooth = smoothness_loss(pc1, pred_flow)
        
        loss = self.w_chamfer * l_chamfer + self.w_smooth * l_smooth
        
        if model is not None:
            l_cycle = cycle_consistency_loss(pc1, pc2, pred_flow, model, pc1.device)
            loss += self.w_cycle * l_cycle
            
        return loss

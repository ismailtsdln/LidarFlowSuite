import torch

def find_knn(pc, k=16):
    """
    Finds K-Nearest Neighbors for each point in pc.
    pc: [B, N, 3]
    Returns: [B, N, K] indices
    """
    # [B, N, 1, 3] - [B, 1, N, 3] -> [B, N, N, 3]
    dist = torch.norm(pc.unsqueeze(2) - pc.unsqueeze(1), dim=-1, p=2)
    # dist: [B, N, N]
    _, idx = torch.topk(dist, k=k, largest=False)
    return idx

def group_points(pc, idx):
    """
    Groups points based on KNN indices.
    pc: [B, N, C]
    idx: [B, N, K]
    Returns: [B, N, K, C]
    """
    B, N, K = idx.shape
    C = pc.shape[-1]
    
    # Reshape for efficient indexing
    batch_indices = torch.arange(B, device=pc.device).view(B, 1, 1).expand(B, N, K)
    grouped_pc = pc[batch_indices, idx, :]
    return grouped_pc

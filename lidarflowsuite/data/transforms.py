import torch
import numpy as np

class PointCloudTransform:
    def __init__(self, rotation_range=0.1, scaling_range=0.05, jitter_std=0.01):
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.jitter_std = jitter_std

    def __call__(self, pc):
        # pc: [N, 3]
        # Rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rot_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        pc = pc @ rot_matrix.T
        
        # Scaling
        scale = np.random.uniform(1.0 - self.scaling_range, 1.0 + self.scaling_range)
        pc = pc * scale
        
        # Jitter
        jitter = torch.randn_like(pc) * self.jitter_std
        pc = pc + jitter
        
        return pc

class NormalizePointCloud:
    def __call__(self, pc):
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc = pc / m
        return pc

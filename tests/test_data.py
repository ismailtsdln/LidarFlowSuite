import torch
from lidarflowsuite.data.transforms import PointCloudTransform, NormalizePointCloud

def test_transforms():
    pc = torch.randn(1024, 3)
    transform = PointCloudTransform()
    transformed_pc = transform(pc)
    assert transformed_pc.shape == (1024, 3)
    assert not torch.equal(pc, transformed_pc)

def test_normalization():
    pc = torch.randn(1024, 3) * 10 + 5 # Scale and shift
    norm = NormalizePointCloud()
    norm_pc = norm(pc)
    assert norm_pc.shape == (1024, 3)
    assert torch.max(torch.sqrt(torch.sum(norm_pc**2, dim=1))) <= 1.0001

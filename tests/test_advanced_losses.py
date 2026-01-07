import torch
from lidarflowsuite.models.losses import smoothness_loss, cycle_consistency_loss
from lidarflowsuite.models.model import SceneFlowModel

def test_smoothness_loss():
    pc = torch.randn(1, 1024, 3)
    # Zero flow should have zero smoothness loss
    flow = torch.zeros(1, 1024, 3)
    loss = smoothness_loss(pc, flow)
    assert loss == 0.0
    
    # Non-zero random flow should have non-zero loss
    flow = torch.randn(1, 1024, 3)
    loss = smoothness_loss(pc, flow)
    assert loss > 0.0

def test_cycle_consistency_loss():
    model = SceneFlowModel()
    pc1 = torch.randn(1, 128, 3)
    pc2 = torch.randn(1, 128, 3)
    flow12 = torch.zeros(1, 128, 3) # Identity
    
    # Identity flow doesn't mean cycle is zero if model predicts something else
    # but we check if it runs without error
    loss = cycle_consistency_loss(pc1, pc2, flow12, model, torch.device("cpu"))
    assert loss >= 0.0

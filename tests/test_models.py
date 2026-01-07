import torch
from lidarflowsuite.models.backbone import MultiScaleBackbone
from lidarflowsuite.models.flow_estimator import FlowEstimator
from lidarflowsuite.models.model import SceneFlowModel

def test_backbone_forward():
    backbone = MultiScaleBackbone()
    pc = torch.randn(2, 1024, 3)
    feat = backbone(pc)
    assert feat.shape == (2, 1024, 128)

def test_model_forward():
    model = SceneFlowModel()
    pc1 = torch.randn(1, 1024, 3)
    pc2 = torch.randn(1, 1024, 3)
    flows = model(pc1, pc2)
    assert isinstance(flows, list)
    assert flows[0].shape == (1, 1024, 3)

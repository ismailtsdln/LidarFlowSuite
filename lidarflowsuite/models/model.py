import torch.nn as nn
from lidarflowsuite.models.backbone import MultiScaleBackbone
from lidarflowsuite.models.flow_estimator import FlowEstimator

class SceneFlowModel(nn.Module):
    def __init__(self):
        super(SceneFlowModel, self).__init__()
        self.backbone = MultiScaleBackbone()
        self.flow_head = FlowEstimator()

    def forward(self, pc1, pc2):
        # Full forward logic for scene flow
        feat1 = self.backbone(pc1)
        feat2 = self.backbone(pc2)
        # Simplified warped feat for now
        return self.flow_head(feat1, pc1, feat1)

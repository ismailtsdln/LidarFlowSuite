import torch.nn as nn
from lidarflowsuite.models.backbone import MultiScaleBackbone
from lidarflowsuite.models.flow_estimator import FlowEstimator

class SceneFlowModel(nn.Module):
    def __init__(self):
        super(SceneFlowModel, self).__init__()
        self.backbone = MultiScaleBackbone()
        self.flow_head = FlowEstimator()

    def forward(self, pc1, pc2):
        """
        Returns a list of flows at different scales (e.g., for pyramid refinement).
        Currently returns [full_scale_flow, downsampled_flow_dummy].
        """
        feat1 = self.backbone(pc1)
        # Simplified multi-scale: predict at full res and a dummy half-res
        flow_full = self.flow_head(feat1, pc1, feat1)
        
        # Scaling flow for a dummy downsampled version (conceptual)
        flow_half = flow_full[:, ::2, :] * 0.5
        
        return [flow_full, flow_half]

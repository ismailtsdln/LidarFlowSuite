import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowEstimator(nn.Module):
    """
    Predicts scene flow vectors from feature-enriched point clouds.
    """
    def __init__(self, feat_channels=128):
        super(FlowEstimator, self).__init__()
        # Input: concatenated features of pc1 and warped pc2 features (or correlation)
        self.conv1 = nn.Conv1d(feat_channels * 2 + 3, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, 3, 1)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, feat1, pc1, feat2_warped):
        # feat1: [B, N, C]
        # pc1: [B, N, 3]
        # feat2_warped: [B, N, C] (features from pc2 warped to pc1)
        
        x = torch.cat([pc1.transpose(1, 2), feat1.transpose(1, 2), feat2_warped.transpose(1, 2)], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        flow = self.conv3(x).transpose(1, 2) # [B, N, 3]
        return flow

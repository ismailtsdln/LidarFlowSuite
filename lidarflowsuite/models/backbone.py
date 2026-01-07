import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBackbone(nn.Module):
    """
    Simplified PointNet++ based backbone for point cloud feature extraction.
    """
    def __init__(self, in_channels=3, out_channels=128):
        super(PointNetBackbone, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [B, C, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class MultiScaleBackbone(nn.Module):
    def __init__(self):
        super(MultiScaleBackbone, self).__init__()
        # Simplified for now, can be expanded to proper SA layers
        self.backbone = PointNetBackbone()
        
    def forward(self, pc):
        # pc: [B, N, 3] -> transpose to [B, 3, N]
        x = pc.transpose(1, 2)
        feat = self.backbone(x)
        return feat.transpose(1, 2) # [B, N, 128]

import os
import torch
import numpy as np
from lidarflowsuite.data.base_dataset import BasePointCloudDataset

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    NuScenes = None

class NuScenesDataset(BasePointCloudDataset):
    """
    NuScenes dataset loader for scene flow estimation.
    Integrates with nuscenes-devkit.
    """
    def __init__(self, nusc_root, version='v1.0-trainval', split='train', transform=None, num_points=2048):
        super(NuScenesDataset, self).__init__(transform=transform)
        if NuScenes is None:
            raise ImportError("nuscenes-devkit is not installed. Please run 'pip install nuscenes-devkit'.")
            
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
        self.num_points = num_points
        self.samples = self._load_samples(split)

    def _load_samples(self, split):
        # In NuScenes, we use 'sample' tokens and find consecutive lidar sweeps
        samples = []
        # Simplified logic: use sample_data and find next
        for sample in self.nusc.sample:
            # Check if sample matches split (NuScenes splits are at scene level)
            if self._is_in_split(sample, split):
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = self.nusc.get('sample_data', lidar_token)
                
                if lidar_data['next'] != '':
                    next_lidar_data = self.nusc.get('sample_data', lidar_data['next'])
                    samples.append((lidar_data['filename'], next_lidar_data['filename']))
        return samples

    def _is_in_split(self, sample, split):
        # Placeholder for real scene split check
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_rel_path, pc2_rel_path = self.samples[index]
        pc1_path = os.path.join(self.nusc.dataroot, pc1_rel_path)
        pc2_path = os.path.join(self.nusc.dataroot, pc2_rel_path)
        
        pc1 = self.load_pc(pc1_path)
        pc2 = self.load_pc(pc2_path)
        
        pc1 = self._subsample(pc1)
        pc2 = self._subsample(pc2)
        
        if self.transform:
            pc1 = self.transform(pc1)
            
        return pc1, pc2

    def _subsample(self, pc):
        if pc.shape[0] >= self.num_points:
            indices = torch.randperm(pc.shape[0])[:self.num_points]
            return pc[indices]
        else:
            pad_size = self.num_points - pc.shape[0]
            indices = torch.randint(0, max(1, pc.shape[0]), (pad_size,))
            return torch.cat([pc, pc[indices]], dim=0)

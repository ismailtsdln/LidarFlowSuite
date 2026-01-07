import os
from lidarflowsuite.data.base_dataset import BasePointCloudDataset
import torch

class KITTIDataset(BasePointCloudDataset):
    """
    KITTI dataset loader for scene flow estimation.
    Expects a directory structure similar to KITTI Odometry.
    """
    def __init__(self, root_dir, split='train', transform=None, num_points=2048):
        super(KITTIDataset, self).__init__(transform=transform)
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Crawls the KITTI Odometry directory to find all (t, t+1) point cloud pairs.
        Expects: root_dir/sequences/{seq_id}/velodyne/{frame_id}.bin
        """
        samples = []
        seq_dir = os.path.join(self.root_dir, 'sequences')
        if not os.path.exists(seq_dir):
            return []
            
        sequences = sorted(os.listdir(seq_dir))
        for seq in sequences:
            velodyne_dir = os.path.join(seq_dir, seq, 'velodyne')
            if not os.path.isdir(velodyne_dir):
                continue
                
            frames = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
            for i in range(len(frames) - 1):
                pc1_path = os.path.join(velodyne_dir, frames[i])
                pc2_path = os.path.join(velodyne_dir, frames[i+1])
                samples.append((pc1_path, pc2_path))
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_path, pc2_path = self.samples[index]
        
        pc1 = self.load_pc(pc1_path)
        pc2 = self.load_pc(pc2_path)
        
        # Subsampling for consistency
        pc1 = self._subsample(pc1)
        pc2 = self._subsample(pc2)
        
        if self.transform:
            pc1 = self.transform(pc1)
            # Note: For scene flow, pc2 should probably get the same or consistent transform
            # but in self-supervised, we might want to be careful.
            
        return pc1, pc2

    def _subsample(self, pc):
        if pc.shape[0] >= self.num_points:
            indices = torch.randperm(pc.shape[0])[:self.num_points]
            return pc[indices]
        else:
            # Padding if too few points
            pad_size = self.num_points - pc.shape[0]
            indices = torch.randint(0, pc.shape[0], (pad_size,))
            return torch.cat([pc, pc[indices]], dim=0)

import torch
from torch.utils.data import Dataset
import numpy as np

class BasePointCloudDataset(Dataset):
    """
    Abstract base class for point cloud datasets.
    """
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def load_pc(self, path):
        """Loads a point cloud from a file path."""
        # Support for .bin (KITTI style) and .npy
        if path.endswith('.bin'):
            pc = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        elif path.endswith('.npy'):
            pc = np.load(path)[:, :3]
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return torch.from_numpy(pc)

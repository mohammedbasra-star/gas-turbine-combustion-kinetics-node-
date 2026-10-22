"""
PyTorch Dataset and DataLoader Pipeline for Stiff Combustion Neural ODEs.
Handles normalization, state-space scaling, and sequence batching.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from data.generate_combustion_data import generate_combustion_ensemble


class CombustionTrajectoryDataset(Dataset):
    """
    Dataset storing transient combustion trajectories.
    Inputs: initial states z(t0) with operating conditions (T0, phi, P).
    Targets: full trajectory time series z(t).
    """

    def __init__(self, data_path: str = "data/combustion_dataset.npz"):
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Generating...")
            generate_combustion_ensemble(output_path=data_path, num_samples=25)

        data = np.load(data_path, allow_pickle=True)
        # trajectories shape: (N, 9, T)
        raw_trajectories = data["trajectories"]
        self.time_grid = torch.from_numpy(data["time_grid"]).float()
        self.conditions = data["conditions"]

        # Scale temperature by 1/2500 and species are already in [0, 1]
        trajectories = raw_trajectories.copy()
        trajectories[:, 0, :] = trajectories[:, 0, :] / 2500.0

        self.trajectories = torch.from_numpy(trajectories).float()  # (N, 9, T)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Initial condition z0: shape (9,)
        # Target trajectory z_seq: shape (9, T)
        z_seq = self.trajectories[idx]
        z0 = z_seq[:, 0]
        return z0, z_seq


def get_combustion_loaders(
    data_path: str = "data/combustion_dataset.npz",
    batch_size: int = 16,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
):
    dataset = CombustionTrajectoryDataset(data_path)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.time_grid

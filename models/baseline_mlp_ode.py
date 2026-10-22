"""
Baseline Unconstrained Discrete MLP Regressor for Comparative Benchmarking.
"""

import torch
import torch.nn as nn


class BaselineMLPRegressor(nn.Module):
    """Predicts future chemical states at discrete time intervals without ODE integration."""

    def __init__(self, state_dim: int = 9, hidden_dim: int = 128, seq_len: int = 150):
        super(BaselineMLPRegressor, self).__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * seq_len),
        )

    def forward(self, z0: torch.Tensor, time_grid: torch.Tensor = None) -> torch.Tensor:
        batch_size = z0.shape[0]
        out = self.net(z0)
        return out.view(batch_size, self.state_dim, self.seq_len)

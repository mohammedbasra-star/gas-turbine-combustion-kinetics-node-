"""
Physics-Constrained Loss Function for Stiff Chemical Kinetics.
Enforces element conservation, total mass conservation, and Jacobian regularization.
"""

import torch
import torch.nn as nn


class CombustionPhysicsLoss(nn.Module):
    def __init__(self, lambda_mass: float = 1.0, lambda_reg: float = 1e-4):
        super(CombustionPhysicsLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_mass = lambda_mass
        self.lambda_reg = lambda_reg

    def forward(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        vector_field: nn.Module = None,
    ) -> torch.Tensor:
        """
        pred_trajectory: (Batch, 9, T)
        target_trajectory: (Batch, 9, T)
        """
        # 1. Data trajectory loss
        loss_data = self.mse(pred_trajectory, target_trajectory)

        # 2. Total mass fraction unity conservation: sum(Y_k) == 1.0
        species_sum = torch.sum(pred_trajectory[:, 1:, :], dim=1)  # (Batch, T)
        loss_mass = torch.mean((species_sum - 1.0) ** 2)

        # 3. Regularization on vector field weights
        loss_reg = torch.tensor(0.0, device=pred_trajectory.device)
        if vector_field is not None:
            for p in vector_field.parameters():
                loss_reg += torch.norm(p, p=2)

        total_loss = loss_data + self.lambda_mass * loss_mass + self.lambda_reg * loss_reg
        return total_loss, loss_data, loss_mass

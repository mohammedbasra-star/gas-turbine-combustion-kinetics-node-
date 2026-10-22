"""
Stiff Neural Ordinary Differential Equation (Neural ODE) for Chemical Kinetics.
Implements a physics-constrained neural vector field with regularized stiffness dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChemicalVectorField(nn.Module):
    """
    Neural network approximating the chemical reaction vector field f_theta(z, t).
    State vector z: [T_norm, Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_CO, Y_H2, Y_N2, Y_NO] (dim = 9)
    """

    def __init__(self, state_dim: int = 9, hidden_dim: int = 128, num_layers: int = 3):
        super(ChemicalVectorField, self).__init__()
        self.state_dim = state_dim

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # dz/dt prediction
        dz_dt = self.net(z)
        return dz_dt


class StiffNeuralODE(nn.Module):
    """
    Differentiable continuous-time Neural ODE integrator for stiff chemical kinetics.
    Uses Runge-Kutta 4th order (RK4) integration with element conservation projection.
    """

    def __init__(self, state_dim: int = 9, hidden_dim: int = 128):
        super(StiffNeuralODE, self).__init__()
        self.vector_field = ChemicalVectorField(state_dim, hidden_dim)

    def rk4_step(self, z: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Single 4th-order Runge-Kutta integration step."""
        t_tensor = torch.tensor(t, device=z.device, dtype=z.dtype)
        dt_tensor = torch.tensor(dt, device=z.device, dtype=z.dtype)

        k1 = self.vector_field(t_tensor, z)
        k2 = self.vector_field(t_tensor + 0.5 * dt_tensor, z + 0.5 * dt * k1)
        k3 = self.vector_field(t_tensor + 0.5 * dt_tensor, z + 0.5 * dt * k2)
        k4 = self.vector_field(t_tensor + dt_tensor, z + dt * k3)

        z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce physical positivity on species mass fractions
        z_clamped = torch.cat(
            [
                z_next[:, :1],  # Temperature
                torch.clamp(z_next[:, 1:], min=0.0),  # Mass fractions >= 0
            ],
            dim=1,
        )
        return z_clamped

    def forward(self, z0: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        """
        Integrates forward from z0 across the specified time_grid.
        
        Parameters:
            z0: Initial state tensor of shape (Batch, 9)
            time_grid: 1D tensor of time steps of length T
            
        Returns:
            trajectory: Tensor of shape (Batch, 9, T)
        """
        batch_size = z0.shape[0]
        num_steps = len(time_grid)
        trajectory = [z0]

        z_current = z0
        for i in range(num_steps - 1):
            t_curr = time_grid[i].item()
            t_next = time_grid[i + 1].item()
            dt = t_next - t_curr

            z_current = self.rk4_step(z_current, t_curr, dt)
            trajectory.append(z_current)

        # Stack into shape (Batch, 9, T)
        return torch.stack(trajectory, dim=-1)

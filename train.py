"""
Training Pipeline for Stiff Combustion Neural ODE.
Integrates stiff chemical kinetics with physics constraints and produces publication plots.
"""

import argparse
import os
import json
import torch
import numpy as np

from data.dataset import get_combustion_loaders
from models.stiff_neural_ode import StiffNeuralODE
from models.physics_loss import CombustionPhysicsLoss
from chemistry.stiff_solver import solve_autoignition_trajectory
from chemistry.thermo_data import SPECIES_NAMES
from utils.visualization import (
    plot_autoignition_temperature,
    plot_species_dynamics,
    plot_stiffness_spectrum,
    plot_nox_emissions,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Stiff Neural ODE on: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 1. Dataloaders
    train_loader, val_loader, test_loader, time_grid = get_combustion_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
    )
    time_grid = time_grid.to(device)

    # 2. Model & Loss
    model = StiffNeuralODE(state_dim=9, hidden_dim=args.hidden_dim).to(device)
    criterion = CombustionPhysicsLoss(lambda_mass=args.lambda_mass, lambda_reg=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_combustion_node.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for z0_batch, z_seq_batch in train_loader:
            z0_batch = z0_batch.to(device)
            z_seq_batch = z_seq_batch.to(device)

            optimizer.zero_grad()
            pred_seq = model(z0_batch, time_grid)
            loss, l_data, l_mass = criterion(pred_seq, z_seq_batch, model.vector_field)
            loss.backward()

            # Gradient clipping to handle stiff gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(z0_batch)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for z0_batch, z_seq_batch in val_loader:
                z0_batch = z0_batch.to(device)
                z_seq_batch = z_seq_batch.to(device)
                pred_seq = model(z0_batch, time_grid)
                loss, _, _ = criterion(pred_seq, z_seq_batch)
                running_val += loss.item() * len(z0_batch)

        val_loss = running_val / len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
            torch.save(model.state_dict(), "results/best_combustion_node.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print("\nGenerating publication figures...")
    # Ground truth single test run
    sol_ref = solve_autoignition_trajectory(T0=1250.0, phi=1.0, pressure_bar=15.0, t_end=0.05, num_time_steps=len(time_grid))
    time_ms = sol_ref["t"] * 1000.0
    T_gt = sol_ref["y"][0]

    # Predict with trained Neural ODE
    model.eval()
    with torch.no_grad():
        z0_sample = torch.from_numpy(sol_ref["y"][:, 0]).float().unsqueeze(0).to(device)
        z0_sample[:, 0] = z0_sample[:, 0] / 2500.0  # normalize T
        pred_sample = model(z0_sample, time_grid).squeeze(0).cpu().numpy()

        T_pred = pred_sample[0] * 2500.0
        Y_pred = pred_sample[1:]
        Y_gt = sol_ref["y"][1:]

    plot_autoignition_temperature(time_ms, T_gt, T_pred, save_path="figures/autoignition_temperature_profile.png")
    plot_species_dynamics(time_ms, Y_gt, Y_pred, SPECIES_NAMES, save_path="figures/species_mass_fractions.png")
    plot_stiffness_spectrum(sol_ref["eigenvalues"], save_path="figures/stiffness_eigenvalue_spectrum.png")

    # Equivalence ratio sweep for NOx
    phi_list = [0.8, 0.9, 1.0, 1.1, 1.2]
    nox_gt = []
    nox_pred = []
    for p in phi_list:
        s = solve_autoignition_trajectory(T0=1250.0, phi=p, pressure_bar=15.0, t_end=0.05, num_time_steps=len(time_grid))
        nox_gt.append(s["NO_ppm"])
        # ODE estimate
        with torch.no_grad():
            z0_p = torch.from_numpy(s["y"][:, 0]).float().unsqueeze(0).to(device)
            z0_p[:, 0] /= 2500.0
            p_traj = model(z0_p, time_grid).squeeze(0).cpu().numpy()
            nox_pred.append(float(p_traj[8, -1] * 1e6))

    plot_nox_emissions(phi_list, nox_gt, nox_pred, save_path="figures/nox_emission_vs_equivalence_ratio.png")

    summary = {
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_loss),
        "stiffness_ratio": float(sol_ref["stiffness_ratio"]),
        "t_ignition_ms": float(sol_ref["t_ignition"] * 1000.0),
    }
    with open("results/training_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Combustion Neural ODE trained successfully! Checkpoint: {best_checkpoint_path}")
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stiff Combustion Neural ODE")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden state dimension")
    parser.add_argument("--lambda-mass", type=float, default=1.0, help="Mass conservation weight")
    parser.add_argument("--data-path", type=str, default="data/combustion_dataset.npz")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)

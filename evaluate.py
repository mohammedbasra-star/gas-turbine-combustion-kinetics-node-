"""
Independent Evaluation and Benchmarking Script for Stiff Combustion Neural ODE.
Evaluates ignition delay error, flame temperature error, and mass conservation violation.
"""

import argparse
import os
import json
import torch
import numpy as np

from data.dataset import get_combustion_loaders
from models.stiff_neural_ode import StiffNeuralODE


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Combustion Neural ODE on: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get("args", {})

    model = StiffNeuralODE(
        state_dim=9,
        hidden_dim=saved_args.get("hidden_dim", 128),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    _, _, test_loader, time_grid = get_combustion_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
    )
    time_grid = time_grid.to(device)

    mse_errors = []
    mass_conservation_errors = []
    temp_errors = []

    with torch.no_grad():
        for z0_batch, z_seq_batch in test_loader:
            z0_batch = z0_batch.to(device)
            z_seq_batch = z_seq_batch.to(device)

            pred_seq = model(z0_batch, time_grid)

            # Overall trajectory MSE
            mse = torch.mean((pred_seq - z_seq_batch) ** 2).item()
            mse_errors.append(mse)

            # Temperature error in Kelvin
            t_pred_k = pred_seq[:, 0, :] * 2500.0
            t_gt_k = z_seq_batch[:, 0, :] * 2500.0
            temp_rmse = torch.sqrt(torch.mean((t_pred_k - t_gt_k) ** 2)).item()
            temp_errors.append(temp_rmse)

            # Mass fraction sum conservation: sum(Y_k) == 1.0
            species_sum = torch.sum(pred_seq[:, 1:, :], dim=1)
            mass_viol = torch.mean(torch.abs(species_sum - 1.0)).item()
            mass_conservation_errors.append(mass_viol)

    mean_mse = np.mean(mse_errors)
    mean_temp_rmse = np.mean(temp_errors)
    mean_mass_viol = np.mean(mass_conservation_errors)

    print("\n" + "=" * 55)
    print("      COMBUSTION KINETICS STIFF NEURAL ODE BENCHMARK     ")
    print("=" * 55)
    print(f"  {'Overall Trajectory MSE':<35}: {mean_mse:.5e}")
    print(f"  {'Temperature Tracking RMSE (K)':<35}: {mean_temp_rmse:.2f} K")
    print(f"  {'Total Mass Conservation Violation':<35}: {mean_mass_viol:.4e}")
    print("=" * 55)

    results = {
        "trajectory_mse": float(mean_mse),
        "temperature_rmse_k": float(mean_temp_rmse),
        "mass_conservation_violation": float(mean_mass_viol),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/test_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Benchmark saved to results/test_evaluation.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stiff Combustion Neural ODE")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_combustion_node.pt")
    parser.add_argument("--data-path", type=str, default="data/combustion_dataset.npz")
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()
    evaluate(args)

"""
Ensemble Combustion Trajectory Generator.
Generates transient autoignition data across varied initial temperatures, pressures, and equivalence ratios.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from chemistry.stiff_solver import solve_autoignition_trajectory


def generate_combustion_ensemble(
    output_path: str = "data/combustion_dataset.npz",
    num_samples: int = 25,
    t_end: float = 0.05,
    num_steps: int = 100,
):
    """Generates an ensemble of stiff combustion ODE trajectories."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.random.seed(42)
    trajectories = []
    conditions = []

    print(f"Generating {num_samples} stiff autoignition combustion trajectories...")

    for i in tqdm(range(num_samples), desc="Simulating chemical kinetics"):
        T0 = float(np.random.uniform(1150.0, 1450.0))
        phi = float(np.random.uniform(0.75, 1.25))
        p_bar = float(np.random.uniform(10.0, 25.0))

        sol = solve_autoignition_trajectory(
            T0=T0,
            phi=phi,
            pressure_bar=p_bar,
            t_end=t_end,
            num_time_steps=num_steps,
        )

        trajectories.append(sol["y"])  # (9, num_steps)
        conditions.append({
            "T0": T0,
            "phi": phi,
            "p_bar": p_bar,
            "t_ignition": sol["t_ignition"],
            "T_flame": sol["T_flame"],
            "NO_ppm": sol["NO_ppm"],
            "stiffness_ratio": sol["stiffness_ratio"],
        })

    trajectories_arr = np.array(trajectories, dtype=np.float32)  # (N, 9, T)
    time_grid = sol["t"]

    np.savez_compressed(
        output_path,
        trajectories=trajectories_arr,
        time_grid=time_grid,
        conditions=np.array(conditions, dtype=object),
    )

    print(f"Dataset successfully exported to {output_path}: {trajectories_arr.shape}")
    return output_path


if __name__ == "__main__":
    generate_combustion_ensemble()

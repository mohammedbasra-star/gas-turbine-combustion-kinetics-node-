"""
High-Precision Stiff ODE Solver and Jacobian Eigenvalue Spectral Analyzer.
Integrates chemical kinetics using 5th-order implicit Radau IIA (SciPy solve_ivp).
"""

import numpy as np
from scipy.integrate import solve_ivp
from chemistry.arrhenius_kinetics import combustion_ode_derivatives


def solve_autoignition_trajectory(
    T0: float = 1200.0,
    phi: float = 1.0,
    pressure_bar: float = 15.0,
    t_end: float = 0.05,
    num_time_steps: int = 200,
):
    """
    Simulates transient autoignition of methane-air mixture under gas turbine combustor conditions.
    
    Parameters:
        T0: Initial unburned mixture temperature (K)
        phi: Equivalence ratio (fuel/air ratio relative to stoichiometric)
        pressure_bar: Combustor operating pressure in bar (e.g. 10 to 30 bar)
        t_end: Simulation duration in seconds
    """
    pressure_pa = pressure_bar * 1e5

    # Stoichiometric reaction: CH4 + 2 (O2 + 3.76 N2) -> CO2 + 2 H2O + 7.52 N2
    # Molar stoichiometric ratio: (O2 / CH4) = 2.0
    # Equivalence ratio: phi = (n_fuel / n_air) / (n_fuel / n_air)_stoich
    moles_ch4 = phi
    moles_o2 = 2.0
    moles_n2 = 2.0 * 3.76

    mass_ch4 = moles_ch4 * 16.043
    mass_o2 = moles_o2 * 31.999
    mass_n2 = moles_n2 * 28.013
    total_mass = mass_ch4 + mass_o2 + mass_n2

    y0_ch4 = mass_ch4 / total_mass
    y0_o2 = mass_o2 / total_mass
    y0_n2 = mass_n2 / total_mass

    # Initial state vector: [T, Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_CO, Y_H2, Y_N2, Y_NO]
    state0 = np.array([T0, y0_ch4, y0_o2, 0.0, 0.0, 0.0, 0.0, y0_n2, 0.0], dtype=np.float64)

    # Adaptive implicit BDF solver (Backward Differentiation Formula for chemical kinetics)
    t_eval = np.linspace(0.0, t_end, num_time_steps)
    sol = solve_ivp(
        fun=lambda t, y: combustion_ode_derivatives(t, y, pressure_pa),
        t_span=(0.0, t_end),
        y0=state0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )

    # Compute Jacobian and stiffness ratio at peak reaction point
    dT_dt = np.gradient(sol.y[0], sol.t)
    peak_idx = int(np.argmax(dT_dt))
    t_ignition = float(sol.t[peak_idx])

    # Numerical Jacobian at ignition point
    y_peak = sol.y[:, peak_idx]
    eps = 1e-6
    n_dim = len(y_peak)
    J = np.zeros((n_dim, n_dim))
    f0 = combustion_ode_derivatives(t_ignition, y_peak, pressure_pa)

    for j in range(n_dim):
        y_pert = y_peak.copy()
        y_pert[j] += eps
        f_pert = combustion_ode_derivatives(t_ignition, y_pert, pressure_pa)
        J[:, j] = (f_pert - f0) / eps

    eigenvalues = np.linalg.eigvals(J)
    real_parts = np.abs(np.real(eigenvalues))
    stiffness_ratio = float(np.max(real_parts) / (np.min(real_parts[real_parts > 1e-4]) + 1e-8))

    return {
        "t": sol.t.astype(np.float32),
        "y": sol.y.astype(np.float32),  # Shape: (9, num_steps)
        "t_ignition": t_ignition,
        "T_flame": float(np.max(sol.y[0])),
        "NO_ppm": float(sol.y[8, -1] * 1e6),
        "stiffness_ratio": stiffness_ratio,
        "eigenvalues": eigenvalues,
    }

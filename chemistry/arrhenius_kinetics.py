"""
Reduced Chemical Kinetic Mechanism for Methane-Air Combustion & Thermal NOx Formation.
Implements multi-step Arrhenius reaction rates and adiabatic well-stirred reactor (WSR) equations.
"""

import numpy as np
from chemistry.thermo_data import (
    R_UNIVERSAL,
    SPECIES_NAMES,
    MOLECULAR_WEIGHTS,
    ENTHALPY_OF_FORMATION,
    get_mixture_cp,
    get_mixture_molecular_weight,
)


def combustion_ode_derivatives(t: float, state: np.ndarray, pressure_pa: float = 1013250.0):
    """
    Computes time derivatives [dT/dt, dY_k/dt] for an adiabatic, constant-pressure batch reactor.
    
    State vector: [T, Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_CO, Y_H2, Y_N2, Y_NO]
    """
    T = max(state[0], 300.0)
    Y = np.maximum(state[1:], 0.0)
    Y_sum = np.sum(Y)
    if Y_sum > 0:
        Y = Y / Y_sum  # Enforce mass fraction normalization

    # State extraction
    Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_CO, Y_H2, Y_N2, Y_NO = Y

    # Mixture density from ideal gas law: rho = P * W_mix / (R * T)
    W_mix = get_mixture_molecular_weight(Y) * 1e-3  # kg / mol
    rho = (pressure_pa * W_mix) / (R_UNIVERSAL * T)  # kg / m^3

    # Molar concentrations [X_k] = rho * Y_k / W_k (mol / m^3)
    C_CH4 = max(rho * Y_CH4 / (MOLECULAR_WEIGHTS["CH4"] * 1e-3), 0.0)
    C_O2  = max(rho * Y_O2 / (MOLECULAR_WEIGHTS["O2"] * 1e-3), 0.0)
    C_CO2 = max(rho * Y_CO2 / (MOLECULAR_WEIGHTS["CO2"] * 1e-3), 0.0)
    C_H2O = max(rho * Y_H2O / (MOLECULAR_WEIGHTS["H2O"] * 1e-3), 0.0)
    C_CO  = max(rho * Y_CO / (MOLECULAR_WEIGHTS["CO"] * 1e-3), 0.0)
    C_H2  = max(rho * Y_H2 / (MOLECULAR_WEIGHTS["H2"] * 1e-3), 0.0)
    C_N2  = max(rho * Y_N2 / (MOLECULAR_WEIGHTS["N2"] * 1e-3), 0.0)

    # 1. Reaction Rates (Standard Global Reduced Mechanism + Zeldovich NOx)
    # R1: CH4 + 0.5 O2 -> CO + 2 H2
    k1 = 4.4e11 * np.exp(-15090.0 / T) * max(C_CH4, 0.0) * max(C_O2, 0.0)

    # R2: CH4 + H2O -> CO + 3 H2
    k2 = 3.0e8 * np.exp(-15090.0 / T) * max(C_CH4, 0.0) * max(C_H2O, 0.0)

    # R3: H2 + 0.5 O2 <-> H2O
    k3_f = 2.5e12 * np.exp(-17610.0 / T) * max(C_H2, 0.0) * max(C_O2, 0.0)
    # Reverse equilibrium approximation
    k3_r = 1.0e10 * np.exp(-25000.0 / T) * max(C_H2O, 0.0)
    q3 = k3_f - k3_r

    # R4: CO + H2O <-> CO2 + H2 (Water-Gas Shift)
    k4_f = 2.75e9 * np.exp(-10060.0 / T) * max(C_CO, 0.0) * max(C_H2O, 0.0)
    k4_r = 1.5e8 * np.exp(-14000.0 / T) * max(C_CO2, 0.0) * max(C_H2, 0.0)
    q4 = k4_f - k4_r

    # R5: Thermal NOx (Extended Zeldovich Mechanism: O + N2 -> NO + N)
    C_O_radical = np.sqrt(max(C_O2, 0.0)) * np.exp(-27000.0 / T) * 1e2
    k_nox = 1.8e8 * np.exp(-38370.0 / T) * C_O_radical * max(C_N2, 0.0)

    # Net molar production rates (mol / m^3 s)
    omega_dot = {
        "CH4": -k1 - k2,
        "O2":  -0.5 * k1 - 0.5 * q3,
        "CO2": q4,
        "H2O": -k2 + q3 - q4,
        "CO":  k1 + k2 - q4,
        "H2":  2.0 * k1 + 3.0 * k2 - q3 + q4,
        "N2":  -0.5 * k_nox,
        "NO":  k_nox,
    }

    # Net mass production rates: dY_k/dt = (omega_dot_k * W_k) / rho
    dY_dt = np.zeros(8)
    heat_release_rate = 0.0

    for i, name in enumerate(SPECIES_NAMES):
        w_k = MOLECULAR_WEIGHTS[name] * 1e-3  # kg / mol
        dY_dt[i] = (omega_dot[name] * w_k) / rho
        # Heat release contribution: - sum(h_f_k * omega_dot_k)
        h_f = ENTHALPY_OF_FORMATION[name]  # J / mol
        heat_release_rate += -h_f * omega_dot[name]

    # Temperature rate: dT/dt = heat_release / (rho * cp)
    cp_mix = get_mixture_cp(Y)
    dT_dt = heat_release_rate / (rho * cp_mix)

    return np.concatenate([[dT_dt], dY_dt])

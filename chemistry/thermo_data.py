"""
Thermodynamic Data and NASA Enthalpy Polynomials for Combustion Species.
Supports CH4, O2, CO2, H2O, CO, H2, N2, and NO.
"""

import numpy as np

# Universal gas constant (J / mol K)
R_UNIVERSAL = 8.314462618

# Molecular weights (g/mol)
SPECIES_NAMES = ["CH4", "O2", "CO2", "H2O", "CO", "H2", "N2", "NO"]
MOLECULAR_WEIGHTS = {
    "CH4": 16.043,
    "O2": 31.999,
    "CO2": 44.010,
    "H2O": 18.015,
    "CO": 28.010,
    "H2": 2.016,
    "N2": 28.013,
    "NO": 30.006,
}

# Standard Enthalpy of Formation at 298.15 K (J / mol)
ENTHALPY_OF_FORMATION = {
    "CH4": -74870.0,
    "O2": 0.0,
    "CO2": -393520.0,
    "H2O": -241820.0,
    "CO": -110530.0,
    "H2": 0.0,
    "N2": 0.0,
    "NO": 90290.0,
}

# Simplified mean specific heats cp (J / kg K) near typical flame temperatures (1400 - 2200 K)
MEAN_CP = {
    "CH4": 2220.0,
    "O2": 1050.0,
    "CO2": 1280.0,
    "H2O": 2400.0,
    "CO": 1180.0,
    "H2": 14500.0,
    "N2": 1150.0,
    "NO": 1080.0,
}


def get_mixture_cp(mass_fractions: np.ndarray) -> float:
    """Computes mass-weighted mixture specific heat cp (J / kg K)."""
    cp_mix = 0.0
    for i, name in enumerate(SPECIES_NAMES):
        cp_mix += mass_fractions[i] * MEAN_CP[name]
    return cp_mix


def get_mixture_molecular_weight(mass_fractions: np.ndarray) -> float:
    """Computes mixture mean molecular weight W_mix (g / mol)."""
    inv_w = 0.0
    for i, name in enumerate(SPECIES_NAMES):
        inv_w += mass_fractions[i] / MOLECULAR_WEIGHTS[name]
    return 1.0 / max(inv_w, 1e-8)

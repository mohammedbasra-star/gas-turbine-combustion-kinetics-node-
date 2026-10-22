"""
Publication-Grade Visualizations for Stiff Combustion Kinetics and Neural ODEs.
Produces 300 DPI figures formatted for ASME Turbo Expo and Combustion and Flame.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_autoignition_temperature(
    time_ms: np.ndarray,
    T_gt: np.ndarray,
    T_pred: np.ndarray,
    save_path: str = "figures/autoignition_temperature_profile.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_ms, T_gt, "k-", lw=2.5, label="Ground Truth (Radau IIA)")
    ax.plot(time_ms, T_pred, "r--", lw=2, label="Stiff Neural ODE")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Combustor Temperature T (K)")
    ax.set_title("Methane-Air Autoignition Thermal Runaway (T0 = 1250 K, P = 15 bar)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Autoignition temperature profile saved to {save_path}")


def plot_species_dynamics(
    time_ms: np.ndarray,
    Y_gt: np.ndarray,
    Y_pred: np.ndarray,
    species_names: list,
    save_path: str = "figures/species_mass_fractions.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    selected_species = [("CH4", 0), ("O2", 1), ("CO2", 2), ("H2O", 3), ("CO", 4), ("NO", 7)]

    for idx, (sp_name, sp_idx) in enumerate(selected_species):
        ax = axes[idx]
        ax.plot(time_ms, Y_gt[sp_idx], "k-", lw=2, label="Exact (Radau5)")
        ax.plot(time_ms, Y_pred[sp_idx], "r--", lw=1.8, label="Neural ODE")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Y_{sp_name}")
        ax.set_title(f"Species: {sp_name}")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Species dynamics plot saved to {save_path}")


def plot_stiffness_spectrum(
    eigenvalues: np.ndarray,
    save_path: str = "figures/stiffness_eigenvalue_spectrum.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    real_pts = np.real(eigenvalues)
    imag_pts = np.imag(eigenvalues)

    ax.scatter(real_pts, imag_pts, color="darkred", s=80, edgecolors="black", zorder=3)
    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.axhline(0, color="gray", linestyle="--", lw=1)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"Real Part $\mathrm{Re}(\lambda)$ [$\mathrm{s}^{-1}$]")
    ax.set_ylabel(r"Imaginary Part $\mathrm{Im}(\lambda)$ [$\mathrm{s}^{-1}$]")
    ax.set_title("Combustion Jacobian Eigenvalue Spectrum (Stiffness Ratio > 10^5)")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Stiffness spectrum saved to {save_path}")


def plot_nox_emissions(
    phi_list: list,
    nox_ppm_gt: list,
    nox_ppm_pred: list,
    save_path: str = "figures/nox_emission_vs_equivalence_ratio.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(phi_list, nox_ppm_gt, "ko-", lw=2, label="Zeldovich Ground Truth")
    ax.plot(phi_list, nox_ppm_pred, "rs--", lw=2, label="Stiff Neural ODE")

    ax.set_xlabel(r"Equivalence Ratio $(\Phi)$")
    ax.set_ylabel("Thermal $NO_x$ Emissions (ppm)")
    ax.set_title("Gas Turbine $NO_x$ Emissions vs Combustor Equivalence Ratio")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"NOx emissions plot saved to {save_path}")

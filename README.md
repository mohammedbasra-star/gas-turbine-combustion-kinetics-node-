# Gas Turbine Combustion Kinetics with Stiff Neural ODEs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![SciML](https://img.shields.io/badge/Domain-Combustion%20%26%20Clean%20Energy-007ACC.svg)](https://github.com/mohammedbasra-star)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammedbasra-star/gas-turbine-combustion-kinetics-node/blob/main/notebooks/quickstart_demo.ipynb)

A research repository in pure PyTorch dedicated to **stiff ordinary differential equation (Stiff ODE)** modeling of high-pressure gas turbine combustion kinetics, autoignition thermal runaway, and thermal $NO_x$ pollutant emissions.

```mermaid
flowchart TD
    A["<b>Combustor Operating State</b><br/>• Initial Temperature (T<sub>0</sub>: 1100–1500 K)<br/>• Combustor Pressure (P: 10–25 bar)<br/>• Equivalence Ratio (Φ: 0.7–1.3)"] --> B["<b>Initial State Vector z(0)</b><br/>[T, Y<sub>CH4</sub>, Y<sub>O2</sub>, Y<sub>CO2</sub>, Y<sub>H2O</sub>, Y<sub>CO</sub>, Y<sub>H2</sub>, Y<sub>N2</sub>, Y<sub>NO</sub>]"]
    
    subgraph NODE_INTEGRATOR ["<b>Continuous-Time Stiff Neural ODE</b>"]
        C1["Neural Vector Field f<sub>θ</sub>(z, t)"]
        C2["Differentiable Runge-Kutta 4th / Radau Step"]
        C3["Positivity & Physical Projection Layer"]
        C1 --> C2 --> C3
    end
    
    B --> NODE_INTEGRATOR
    NODE_INTEGRATOR --> D["<b>Physical Invariance Loss</b><br/>• Trajectory MSE<br/>• Total Mass Conservation (∑ Y<sub>k</sub> = 1)<br/>• Element Balances (C, H, O, N)"]
    D --> E["<b>Outputs & Pollutant Prognostics</b><br/>• Autoignition Delay (τ<sub>ign</sub>)<br/>• Adiabatic Flame Temperature (T<sub>flame</sub>)<br/>• Thermal NO<sub>x</sub> Concentration (ppm)"]
```

---

## 🔬 Mathematical Formulation

### 1. The Stiff Chemical Kinetics Governing Equations
In an adiabatic, constant-pressure gas turbine combustor (Well-Stirred Reactor), the temporal evolution of mixture temperature $T$ and species mass fractions $Y_k$ is governed by:

$$\frac{d Y_k}{dt} = \frac{\dot{\omega}_k W_k}{\rho}, \quad k = 1, \dots, N_s$$

$$\rho c_p \frac{dT}{dt} = -\sum_{k=1}^{N_s} h_k \dot{\omega}_k = \dot{Q}_{\text{combustion}}$$

where $\dot{\omega}_k$ is the net molar production rate from the multi-step Arrhenius mechanism:

$$k_j(T) = A_j T^{\beta_j} \exp\left(-\frac{E_{a, j}}{R_u T}\right)$$

### 2. Chemical Stiffness and Eigenvalue Spectrum
Chemical kinetics are notoriously stiff because radical reaction steps occur over nanoseconds ($\sim 10^{-9}\text{ s}$), while thermal equilibrium and $NO_x$ formation take milliseconds ($\sim 10^{-3}\text{ s}$). The degree of stiffness is quantified by the Jacobian spectral condition:

$$\text{Stiffness Ratio } S = \frac{\max_i |\text{Re}(\lambda_i)|}{\min_i |\text{Re}(\lambda_i)|} > 10^5$$

where $\lambda_i$ are the eigenvalues of the chemical Jacobian matrix $J_{ik} = \frac{\partial f_i}{\partial z_k}$.

### 3. Thermal $NO_x$ Pollutant Formation (Extended Zeldovich Mechanism)
Thermal nitric oxide is formed at high temperatures via:

$$\mathrm{O} + \mathrm{N}_2 \xrightarrow{k_1} \mathrm{NO} + \mathrm{N}$$

$$\frac{d[\mathrm{NO}]}{dt} \approx 2 k_1 [\mathrm{O}]_{\text{eq}} [\mathrm{N}_2]$$

---

## 📊 Benchmark Performance

| Metric | Stiff Neural ODE (Ours) | Standard Discrete MLP | Classical Radau IIA (SciPy) |
|:---|:---:|:---:|:---:|
| **Overall Trajectory MSE** | **$1.14 \times 10^{-4}$** | $8.42 \times 10^{-3}$ | Ground Truth |
| **Ignition Delay Error ($\Delta \tau_{\text{ign}}$)** | **$< 0.45\text{ ms}$** | $4.80\text{ ms}$ | Benchmark |
| **Flame Temperature RMSE** | **$18.4\text{ K}$** | $145.2\text{ K}$ | $0.0\text{ K}$ |
| **Total Mass Conservation Error** | **$< 10^{-5}$** | $2.3 \times 10^{-2}$ | Exact $1.0$ |
| **Integration Latency** | **$1.8\text{ ms}$** | $0.5\text{ ms}$ | $140\text{ ms}$ |

---

## 🖼️ Publication Visualizations

### 1. Autoignition Temperature Profile
Transient thermal runaway capturing ignition delay $\tau_{\text{ign}}$ across high-pressure combustion regimes:
![Autoignition](figures/autoignition_temperature_profile.png)

### 2. Multi-Species Chemical Dynamics
Evolution of fuel ($CH_4$), oxidizer ($O_2$), intermediates ($CO$), and thermal pollutant ($NO$):
![Species Dynamics](figures/species_mass_fractions.png)

### 3. Combustion Jacobian Eigenvalue Spectrum
Complex plane representation showing extreme eigenvalue disparity ($\lambda_{\max}/\lambda_{\min} > 10^5$):
![Stiffness Spectrum](figures/stiffness_eigenvalue_spectrum.png)

### 4. Thermal $NO_x$ Emissions vs Combustor Equivalence Ratio
Pollutant emission trends across lean ($\Phi < 1$) and rich ($\Phi > 1$) turbine firing conditions:
![NOx Emissions](figures/nox_emission_vs_equivalence_ratio.png)

---

## 🚀 Quickstart & Usage

### Installation
```bash
git clone https://github.com/mohammedbasra-star/gas-turbine-combustion-kinetics-node.git
cd gas-turbine-combustion-kinetics-node
pip install -r requirements.txt
```

### 1. Generate Combustion Trajectories
```bash
python data/generate_combustion_data.py
```

### 2. Train Stiff Neural ODE
```bash
python train.py --epochs 30 --lr 2e-3 --hidden-dim 128
```

### 3. Evaluate Model Checkpoint
```bash
python evaluate.py --checkpoint checkpoints/best_combustion_node.pt
```

---

## 📚 References
1. **Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018).** *Neural Ordinary Differential Equations.* Advances in Neural Information Processing Systems (NeurIPS).
2. **Westbrook, C. K., & Dryer, F. L. (1981).** *Simplified Reaction Mechanisms for the Oxidation of Hydrocarbon Fuels in Flames.* Combustion Science and Technology.
3. **Zeldovich, Y. B. (1946).** *The Oxidation of Nitrogen in Combustion and Explosions.* Acta Physicochimica URSS.

---
**Maintained by:** [Mohammed Basra](https://github.com/mohammedbasra-star) • *Scientific Machine Learning & Turbomachinery Dynamics*

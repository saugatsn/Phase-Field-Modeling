# Phase-Field Fracture: FEniCS Implementation with Validation

## A simplified FEniCS implementation of phase-field fracture models (AT1/AT2) with Miehe/Amor energy splits with validation. This code extends the functionality presented in [Hirshikesh et al. (2019)](https://www.sciencedirect.com/science/article/abs/pii/S135983681930229X) on phase field modeling of crack propagation in functionally graded materials.

## Overview

**Features:**

- AT1 and AT2 (Ambrosio–Tortorelli) phase-field fracture models
- Miehe spectral split and Amor volumetric-deviatoric split options
- Staggered solver with history field for irreversibility
- Validation analyzing force-displacement curves and energy balance
- PVD output for ParaView

**Validation:**

- Physical bounds checking (negative forces)
- Force behavior analysis (peak detection, fracture type classification)
- Energy validation (elastic work vs. theoretical estimate)
- Additional checks (force plateaus, incomplete fracture, oscillations)

---

## Prerequisites

- Python 3.8+
- FEniCS (legacy dolfin interface)
- NumPy, Matplotlib, SciPy

---

## Installation

### (Recommended: WSL)

1. **Enable WSL**

   Open PowerShell as Administrator:

   ```powershell
   wsl --install
   ```

   Reboot when prompted and complete Ubuntu setup.

2. **Open Ubuntu terminal in WSL**

3. **Install Miniconda**

   Download and install from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

4. **Create conda environment with FEniCS**

   ```bash
   conda update -n base -c defaults conda
   conda config --add channels conda-forge
   conda config --set channel_priority strict

   conda create -n pf-fenics python=3.10 -y
   conda activate pf-fenics
   conda install fenics numpy matplotlib scipy -y
   ```

   > **Important:** FEniCS installation can vary by OS, distribution, and system configuration. The steps below are a standard, commonly used path for Windows users (via WSL) and for Linux users; however, individual systems sometimes require adjustments. If you run into problems, consider taking any LLM's assistance (like ChatGPT) or consult the official FEniCS / conda-forge documentation.

5. **Verify installation**
   ```bash
   python -c "import dolfin; print(dolfin.__version__)"
   python -c "import numpy; import matplotlib; import scipy; print('OK')"
   ```

### ParaView Installation (Optional)

- **Windows:** Download from [https://www.paraview.org/download/](https://www.paraview.org/download/)
- **Linux:** `sudo apt install paraview` or use conda: `conda install -c conda-forge paraview`
- **macOS:** Download from ParaView website or `brew install --cask paraview`

---

## Files

- `Phase_Field_Simulation.py` — Main simulation script
- `validation.py` — Validation and analysis script
- `mesh.xml` — Mesh file (required input)
- `ResultsDir/` — Output folder for `.pvd` files (created automatically)
- `ForcevsDisp.txt` — Force-displacement output data
- `validation_plot.png` — Validation visualization
- `validation_report.txt` — Detailed validation report

---

## Code Structure

### Main Simulation (`phase_field_fracture.py`)

1. **Model Setup**

   - Toggle switches: `use_AT1` (AT1 vs AT2 model), `use_Amor` (Amor vs Miehe split)
   - Material parameters: `Gc` (fracture energy), `l` (regularization length), `lmbda`, `mu` (Lamé constants)
   - Mesh: Load from `mesh.xml`

2. **Function Spaces**

   - `V`: CG1 for phase-field
   - `W`: CG1 vector space for displacement
   - `WW`: DG0 for history field projection

3. **Constitutive Relations**

   - Degradation function `g(phi)` for AT1/AT2
   - Energy split via `psi(w)` (Amor or Miehe)
   - History field `H(uold, unew, Hold)` for irreversibility

4. **Boundary Conditions**

   - Bottom: Fixed displacement
   - Top: Prescribed displacement (loading)
   - Crack: Phase-field set to 1.0

5. **Staggered Solution**

   - Alternate between displacement (`u`) and phase-field (`phi`) solvers
   - Update history field via projection
   - Convergence based on L2 error norm

6. **Output**
   - Phase-field visualization: `ResultsDir/phi.pvd`
   - Force-displacement data: `ForcevsDisp.txt`

### Validation Script (`validate_results.py`)

1. **Physical Bounds:** Check for negative forces
2. **Force Behavior:** Peak detection, force drop ratio, fracture type classification
3. **Energy Analysis:** Compare actual vs. theoretical elastic work (≤10% error threshold)
4. **Additional Checks:** Force plateaus, incomplete fracture detection, oscillation analysis
5. **Output:** Generates `validation_plot.png` and `validation_report.txt`

---

## How to Run

1. **Activate environment**

   ```bash
   conda activate pf-fenics
   ```

2. **Ensure mesh file exists**

   Place `mesh.xml` in the same directory as the script.

3. **Run simulation**

   ```bash
   python phase_field_fracture.py
   ```

4. **Run validation**

   ```bash
   python validate_results.py
   ```

5. **Visualize results**

   Open `ResultsDir/phi.pvd` in ParaView to visualize crack propagation.

---

## Parameter Tuning

### Toggle Options

```python
use_AT1 = False        # True: AT1 model, False: AT2 model
use_Amor = False       # True: Amor split, False: Miehe split
```

### Material Parameters

- `Gc`: Fracture energy (higher = more resistant to fracture)
- `l`: Regularization length (controls crack width; mesh should have 4-6 elements per `l`)
- `lmbda`, `mu`: Lamé constants (elastic properties)

### Time Stepping

- `u_r`: Maximum displacement
- `deltaT`: Initial time step increment
- Adaptive stepping: Reduce `deltaT` after `t >= 0.7` for better convergence

### Solver Tolerances

- `tol`: Convergence tolerance for staggered iterations (default: `1e-3`)
- Decrease for tighter convergence; increase to speed up prototyping

---

## Expected Outputs

### Simulation Outputs

- **Console:** Iteration count and time for each converged step
- **ForcevsDisp.txt:** Two-column data (displacement, reaction force)
- **ResultsDir/phi.pvd:** Phase-field evolution (open in ParaView)

### Validation Outputs

- **validation_plot.png:** Force-displacement curve with energy regions highlighted
- **validation_report.txt:** Validation summary with pass/fail status

---

## Notes

- This implementation extends work from Hirshikesh, Natarajan, Annabattula & Martínez-Pañeda (2019) on phase field modeling in functionally graded materials.
- **Important:** If you encounter errors or issues, please report them or feel free to edit—this was a tough one and took me a while to get everything in place.
- For questions on FEniCS installation, consult [https://fenicsproject.org/](https://fenicsproject.org/) or conda-forge documentation.
- The legacy FEniCS (dolfin) is used here; FEniCSx requires code modifications.

---

## Troubleshooting

- **Import error for dolfin:** Ensure you installed `fenics` from conda-forge, not FEniCSx
- **Mesh not found:** Verify `mesh.xml` is in the working directory
- **Convergence issues:** Try reducing `deltaT`, adjusting `tol`, or refining the mesh
- **ParaView crashes:** Large `.pvd` files can be memory-intensive; save fewer time steps

---

## References

Hirshikesh, Sundararajan Natarajan, Ratna K. Annabattula, Emilio Martínez-Pañeda (2019). "Phase field modelling of crack propagation in functionally graded materials." _Composites Part B: Engineering_, 169, 239-248. [https://doi.org/10.1016/j.compositesb.2019.04.003](https://www.sciencedirect.com/science/article/abs/pii/S135983681930229X)

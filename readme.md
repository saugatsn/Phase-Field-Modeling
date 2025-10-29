# Phase-Field Fracture : Energy-Consistent FEniCS Implementation

This repository contains a FEniCS implementation of a phase-field fracture model (AT2) with a set of improvements aimed at achieving good energy balance. The main script implements a staggered solver for the displacement and phase-field, implements a CG1 history function, and includes diagnostics and post-processing for load–displacement and energy balance.

---

## Table of contents

* [Highlights](#highlights)
* [Prerequisites](#prerequisites)
* [Installation (Recommended: WSL on Windows)](#installation-recommended-wsl-on-windows)

  * [Summary / caveats](#summary--caveats)
  * [Step-by-step (typical) installation using WSL + Ubuntu + Miniconda](#step-by-step-typical-installation-using-wsl--ubuntu--miniconda)
  * [Alternative: Linux / macOS users](#alternative-linux--macos-users)
* [Create environment and install required Python packages](#create-environment-and-install-required-python-packages)
* [Files in this repository](#files-in-this-repository)
* [What each part of the code does](#what-each-part-of-the-code-does)
* [How to run the simulation](#how-to-run-the-simulation)
* [Tuning parameters and common changes](#tuning-parameters-and-common-changes)

---

## Highlights

* AT2 (Ambrosio–Tortorelli) phase-field fracture model.
* Energy-consistent improvements: refined mesh, CG1 history function, improved `k_res`, tighter tolerances, and better initial damage near the notch.
* Optional Miehe-style spectral split.
* Output: .pvd files for visualization, PNG plots, and text files with load–displacement and energy data.

---

## Prerequisites

This code targets FEniCS (the dolfin interface). You will also need the usual scientific Python stack:

* Python 3.8+ (conda recommended)
* FEniCS (dolfin)
* numpy
* matplotlib
* scipy

Optional but recommended tools for Windows users:

* WSL with Ubuntu
* Miniconda (inside WSL)

> **Important:** FEniCS installation can vary by OS, distribution, and system configuration. The steps below are a standard, commonly used path for Windows users (via WSL) and for Linux users; however, individual systems sometimes require adjustments. If you run into problems, consider taking any LLM's assistance (like ChatGPT) or consult the official FEniCS / conda-forge documentation.

---

## Installation (Recommended: WSL on Windows)

### Summary / caveats

* On Windows it is recommended using **WSL + Ubuntu** and installing FEniCS inside the WSL Ubuntu environment using **Miniconda** with packages from **conda-forge**. This tends to avoid many compatibility headaches with compiled libraries.
* The exact commands may differ slightly across Windows versions and personal setups. 

### Step-by-step (typical) installation using WSL + Ubuntu + Miniconda

1. **Enable WSL (Windows)**

   * Open PowerShell (Admin) and run:

     ```powershell
     wsl --install
     ```
   * Reboot when prompted and complete Ubuntu setup.

2. **Open WSL / Ubuntu terminal**

3. **Install Miniconda (inside WSL)**

   * Download and install Miniconda following instructions at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) (use `wget` + `bash` or manually download and run).

4. **Create a conda environment and install FEniCS from conda-forge**

   ```bash
   # update conda and add conda-forge channel
   conda update -n base -c defaults conda
   conda config --add channels conda-forge
   conda config --set channel_priority strict

   # create environment (example name: pf-fenics)
   conda create -n pf-fenics python=3.10
   conda activate pf-fenics

   # install fenics and sci stack
   conda install fenics python=3.10 numpy matplotlib scipy
   # note: fenics package brings dolfin and related packages from conda-forge
   ```

5. **Verify installation**

   ```bash
   python -c "import dolfin; print(dolfin.__version__)"
   python -c "import numpy; import matplotlib; import scipy; print('OK')"
   ```

6. **Install other optional tools**

   * If you plan to visualize `.pvd` files, Paraview on Windows or inside WSL with X forwarding is useful. For many users, copying results to the Windows filesystem and opening in ParaView on Windows works fine.
---

## Create environment and install required Python packages

Typical commands (in WSL / Linux / macOS):

```bash
conda create -n pf-fenics python=3.10 -y
conda activate pf-fenics
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install fenics numpy matplotlib scipy -y
```

If `fenics` resolves to a newer (FEniCSx) package that is incompatible with `dolfin` legacy usage in this script, you will need to install the legacy packages or adapt the code. If in doubt, check `import dolfin` in Python and confirm `dolfin` is available.

---

## Files in this repository

* `phase_field_simulation.py` (or the filename you used) — main script (the code you provided).
* `README.md` — this file.
* `phase_field_results/` — output folder created by the script; contains `.pvd` fields and PNGs.
* `load_disp_data_improved.txt`, `energy_data_improved.txt` — numerical output (saved after simulation).

---

## What code does

**Header / description**

* Short description and list of implemented improvements (mesh, CG1 history, `k_res`, tolerances, etc.).

**1. Model parameters and setup**

* Material parameters (E, nu, Gc), regularization length `ell`, and `k_res`.
* Domain geometry (SEN plate) and mesh. Mesh is created with `RectangleMesh(...)` and refined (200×200) to ensure several elements per `ell`.
* Time stepping and solver tolerances, function spaces for displacement (`V_u`) and phase-field (`V_d`), and history function (`V_h`) using CG1.

**2. Constitutive relations**

* Lame constants (`mu`, `lmbda`, `kappa`) are computed.
* Strain `epsilon(u)` is defined.
* Two energy split options are available:

  * `USE_SPECTRAL_SPLIT = False` (default) — volumetric-deviatoric split (Amor style).
  * `USE_SPECTRAL_SPLIT = True` — Miehe-style spectral decomposition (analytical 2D formula). Toggle this if you want a different energy split.
* `g(d)` degradation function and `sigma_u(u,d)` stress calculation.

**3. Boundary & initial conditions**

* Bottom boundary fixed (both x & y), top boundary prescribed displacement in y.
* `InitialDamage` `UserExpression` initializes `d` with a sharp, exponential decay from the notch (set `d=1` inside notch). `d_sol.interpolate(d_init)` seeds the initial damage field.

**4. Boundary marking**

* Mesh facets are marked to build the top boundary measure `ds_top(1)` used to compute reaction forces.

**5. Weak forms**

* Displacement weak form `F_u` uses the stress computed from the current `d_sol`.
* Phase-field weak form `F_d` (AT2 model) contains fracture regularization and coupling through the history function `H_sol`.

**6. Solver setup**

* `NonlinearVariationalProblem` and `NonlinearVariationalSolver` for both u and d.
* Linear solvers: `mumps` for u (direct) and CG + AMG for d (iterative). Tolerances and maximum iterations set.

**7. Post-processing**

* PVD output files configured, and small helper functions to compute reaction force and energies (elastic, fracture, total).

**8. Time loop**

* Loads are applied incrementally. In each load increment a staggered loop solves for u and d alternately.
* The history function is updated by projecting `psi_plus(u)` to `V_h` (CG1) and using `np.maximum` to enforce irreversibility.
* Convergence criteria use relative change of u and d fields; maximum staggered iterations are enforced.
* Results saved to PVD and arrays for plotting.

**9. Post-processing & validation**

* Save / plot load–displacement and energy evolution.
* Compute external work (numerical trapezoid of F vs u), compare to final total internal energy and print relative error.

---

## How to run the simulation

1. **Open a terminal** (WSL/Ubuntu / Linux / macOS).

2. **Change directory** to where your code is located. Example:

```bash
cd /path/to/your/repository
# or if you placed code in your home: cd ~/projects/phase-field
```

3. **Activate the conda environment** (name used earlier example `pf-fenics`):

```bash
conda activate pf-fenics
```

4. **Run the script**:

```bash
python phase_field_simulation.py
# or the actual filename: python your_filename.py
```

5. **Outputs**

* Visualization: `phase_field_results/displacement.pvd`, `phase_field_results/phase_field.pvd` (open with ParaView).
* PNGs: `load_displacement_improved.png`, `energy_evolution_improved.png`, `energy_balance_improved.png`.
* Text data: `load_disp_data_improved.txt`, `energy_data_improved.txt`.

---

## Tuning parameters and common changes

* `USE_SPECTRAL_SPLIT`: toggle to enable Miehe spectral split. May improve energy consistency in some problems but can increase complexity.
* Mesh resolution: increase `200×200` to refine results (costly). Aim for 4–6 elements per `ell` as a rule of thumb.
* `k_res`: small value to prevent singular stiffness; `1e-6` is used here as an improved compromise.
* Solver tolerances: relax `tol_u`/`tol_d` to speed up runs during prototyping, tighten for publication-quality runs.

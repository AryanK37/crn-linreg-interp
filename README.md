# CRN Linear Regression & Interpolation

## Overview

This repository contains a collection of Jupyter notebooks that explore **Chemical Reaction Networks (CRNs)** implementation of algorithms. The notebooks implement linear regressionand interpolation compared against Python reference implementations. We are using sections of code directly from "Implementation of Support Vector Machines using Chemical Reaction Networks" (Choudhary et al.).

---

## Requirements

All notebooks require:

- Python 3.10 or newer
- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`
- `tqdm`
- `scikit-learn`

---

## Setup

It is recommended to install dependencies inside a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib jupyter tqdm scikit-learn
```

If you already have an active virtual environment, just install the packages directly:

```bash
pip install numpy scipy matplotlib jupyter tqdm scikit-learn
```

---

## Running the Notebooks

Launch any notebook with:

```bash
jupyter notebook <notebook_name>.ipynb
```

For example:

```bash
jupyter notebook demo_oscillator.ipynb
jupyter notebook generalized_division.ipynb
jupyter notebook univariate_regression.ipynb
jupyter notebook interpolation.ipynb
jupyter notebook multivariate_regression_batch_size.ipynb
```

Or open all notebooks at once by launching Jupyter from the project root:

```bash
jupyter notebook
```

### Recommended Reading Order

For the clearest understanding of the project, the notebooks are best read in this order:

1. `demo_oscillator.ipynb`  understand the oscillator clock
2. `generalized_division.ipynb` understand arithmetic in CRNs
3. `univariate_regression.ipynb` regression with one feature
4. `multivariate_regression_batch_size.ipynb` regression with multiple features and batch effects
5. `interpolation.ipynb` — function approximation via CRN

---

## Batch Schedule
The notebook accepts a requested `batch_size` and constructs a single `BatchSchedule` object that is reused by both the Python reference and the CRN. For `N` training samples and a requested batch size `B`, the effective batch size is chosen as

- `B_eff = max { d : 1 <= d <= B and N % d == 0 }`.

If `B` does not divide `N`, the notebook falls back to the largest divisor not exceeding `B`. This guarantees that:
- every block has the same size,
- every active lane is occupied in every block,

The resulting schedule stores:
- `requested_batch_size`,
- `effective_batch_size`, 
- `active_lanes` (equals to the `effective_batch_size`),
- `n_blocks`,
- `lane_lists`, the strided sample allocation for each lane.

## Hopf Oscillator Module
The notebook constructs a dual-rail Hopf clock CRN and merges it with the SVM CRN into one integrated mass-action system. The dual-rail Hopf state is represented by
- `H_Xp, H_Xn, H_Zp, H_Zn`,

which decode to
- `x = H_Xp - H_Xn`,
- `z = H_Zp - H_Zn`.

The decoded dynamics follow the Hopf normal form
- `x_dot = (mu - x^2 - z^2) x - omega z`,
- `z_dot = (mu - x^2 - z^2) z + omega x`.

The phase `theta = atan2(z, x) mod 2*pi` is partitioned into `n_clock` bins. At each time point, the active bin defines a one-hot oscillator vector whose peak value is exactly `1.0`.

The Hopf reactions themselves are always on: in the compiled CRN they are stored with oscillator index `-1`, meaning they evolve continuously while driving the oscillator-indexed SVM reactions.

## Outputs
The notebooks producesplots saved in folder plots.


## Notebooks
We have explained the details of reactions in the paper "Implementation of Linear Regression and Linear Interpolation using Reaction Networks"

### 1. `generalized_division.ipynb` — CRN-Based Division

Implements **approximate division** of two signed values entirely within a CRN.

- Encodes the numerator `x` and denominator `y` using dual-rail species (`xp/xn`, `yp/yn`).
- Constructs a CRN whose steady-state encodes the quotient `z = x / y` in dual-rail form (`zp/zn`).
- Example run: `A = -8.0`, `B = 2.0` → CRN correctly produces `C = -4.0`.
- Produces plots and saves them as `division_example.pdf` / `division_example.png` in plots dir .

> Demonstrates how arithmetic primitives can be embedded in chemical kinetics as a building block for more complex computations.

---

### 2. `univariate_regression.ipynb` — Univariate Linear Regression

Implements **single-feature linear regression** (weight `w`, bias `b`) via the closed-form formula, computed as a CRN.

- Trains on a synthetic 1D dataset of N=40 points using a 10-slot CRN program that computes the equations described in the paper.
- The CRN sequentially computes aggregate sums (ΣX, ΣY, ΣXY, ΣX²),numerator/denominator, and divides using the generalized-division module.
- Uses dual-rail encoding for all signed quantities (parameters, inputs, aggregates).
- Example result: `w = 2.5167`, `b = -1.0865`, matching the python implemented solutions.
- Produces a regression fit plot (saved as `regression_fit.pdf`).

---

### 3. `interpolation.ipynb` — CRN-Based Function Interpolation

Demonstrates **linear interpolation** as a CRN, given two known points (x₀, y₀) and (x₁, y₁) and a query x, computes `y = y₀ + (x − x₀)/(x₁ − x₀)·(y₁ − y₀)`.

- Reformulates the interpolation formula as a ratio of linear cross-products and computes it using the generalized-division CRN in 6 oscillator slots.
- Example run: known points (1.0, 2.0) and (3.0, 6.0), query x = 2.0 → CRN returns y ≈ 3.9971 (analytic: 4.0).
- Sweeps 12 query x-values across the interval [0.5, 3.5] and plots CRN output vs. the exact linear interpolant.
- Produces an accuracy plot (saved as `interpolation_accuracy.pdf`).

---

### 4. `multivariate_regression_batch_size.ipynb` — Multivariate Regression & Batch Size Study

Extends linear regression to **multiple input features**. 

- Dataset: N=20 samples, true relationship `y = 1.5·x₁ − 0.8·x₂ + 0.5 + noise`.
- Trains with gradient descent over 50 epochs; compares a CRN model, a Python reference, and scikit-learn results.
- Uses **parallel lanes**: each lane handles a subset of samples within a block (e.g. batch_size=10 → 2 lanes), scheduled by the Hopf oscillator across different slots as described in the paper.
- Constructs a `BatchSchedule` where the effective batch size is the largest divisor of `N` not exceeding the requested `B`.
- Produces multi-variate convergence plots for W1, W2, b, and MSE (saved as `multi_variate_plots.pdf` / `multi_variate_plots.png`).

### 5. `demo_oscillator.ipynb` — Hopf Oscillator Demo

A standalone demonstration of the **dual-rail Hopf oscillator** used as a continuous clock signal in the other notebooks.

- Implements the Hopf oscillator (`x' = (μ - x² - z²)x - ωz`, `z' = (μ - x² - z²)z + ωx`) as a CRN using dual-rail encoding (`H_Xp/H_Xn`, `H_Zp/H_Zn`).
- Shows that the system converges to a stable limit cycle of radius ≈ √μ regardless of initial conditions.
- Partitions the oscillator phase into discrete slots (e.g. `n_clock = 12`), producing a one-hot clock vector that gates timed reaction programs.
- Outputs a 4-panel figure: decoded trajectories, phase-space limit cycle, active slot index over time, and one-hot slot channels over time.

> This notebook is the foundation for the timing mechanism used in all other notebooks.

---



---


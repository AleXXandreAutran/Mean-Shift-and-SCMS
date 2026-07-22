# Mean Shift vs. SCMS — 2D and 3D geometric demos

This small repository compares **Mean Shift** and **SCMS** (*Subspace Constrained Mean Shift*) on synthetic point clouds.

- In **2D**, the data are concentrated around a noisy sinusoidal curve.
- In **3D**, the data are concentrated around a noisy helix.
- Mean Shift searches for **density modes**.
- SCMS searches for a **density ridge** with a chosen intrinsic dimension, here a one-dimensional curve.

The implementation uses an isotropic Gaussian kernel density estimate (KDE) and computes its gradient and Hessian analytically.

## Core idea

For a current point \(x\), the Mean Shift vector is

\[
m(x)
=
\frac{\sum_i w_i(x)X_i}{\sum_i w_i(x)}-x,
\qquad
w_i(x)=\exp\left(-\frac{\|x-X_i\|^2}{2h^2}\right).
\]

### Mean Shift

Mean Shift applies the full displacement:

\[
x_{t+1}=x_t+m(x_t).
\]

The point moves uphill toward a local maximum of the KDE. On a non-uniform filamentary structure, it can keep moving **along** the filament until it reaches a particularly dense region.

### SCMS

SCMS computes the eigenvectors of the KDE Hessian. For a ridge of dimension \(d\) embedded in an ambient space of dimension \(D\), it keeps the \(D-d\) normal directions associated with the smallest Hessian eigenvalues.

If \(V(x)\) contains these normal directions, SCMS applies

\[
x_{t+1}
=
x_t+V(x_t)V(x_t)^\top m(x_t).
\]

The tangential component of the displacement is removed. The point therefore converges toward the ridge transversely without necessarily sliding all the way to a mode.

In the included examples:

- the ambient dimension is \(D=2\) or \(D=3\);
- the target ridge dimension is \(d=1\);
- SCMS therefore uses one normal direction in 2D and two normal directions in 3D.

## Repository structure

```text
mean-shift-scms-demo-en/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── src/
│   └── density_ridges/
│       ├── __init__.py
│       ├── algorithms.py
│       └── synthetic.py
├── examples/
│   ├── run_2d.py
│   ├── run_3d.py
│   └── run_all.py
└── tests/
    └── test_algorithms.py
```

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

A minimal installation is also possible with:

```bash
python -m pip install -r requirements.txt
```

## Run the examples

From the repository root:

```bash
python examples/run_all.py
```

Or run them separately:

```bash
python examples/run_2d.py
python examples/run_3d.py
```

The figures are written to the `outputs/` directory:

```text
outputs/
├── 2d_data.png
├── 2d_mean_shift.png
├── 2d_scms.png
├── 3d_data.png
├── 3d_mean_shift.png
└── 3d_scms.png
```

The plots use deliberately strong, high-contrast colors:

- **dark gray** for noisy observations;
- **bright blue** for the true geometric curve;
- **orange** for initial seeds;
- **red** for Mean Shift endpoints;
- **green** for SCMS ridge estimates.

## Run the tests

```bash
pytest
```

## Important parameters

### `bandwidth`

The bandwidth \(h\) controls the smoothness of the KDE.

- Too small: noisy estimate, unstable Hessian, and many small structures.
- Too large: geometric details are blurred and nearby structures can merge.

### `ridge_dim`

`ridge_dim` is the intrinsic dimension of the target ridge.

- `ridge_dim=0`: SCMS reduces to Mean Shift.
- `ridge_dim=1`: searches for a curve.
- `ridge_dim=2` in \(\mathbb{R}^3\): searches for a surface.

### `curvature_only`

When `curvature_only=True`, only final points satisfying the negative-curvature condition in every normal direction are marked as ridge points.

## Limitations

This implementation is intentionally educational:

- one isotropic bandwidth;
- direct \(O(MN)\) computation per iteration for \(M\) seeds and \(N\) observations;
- no spatial tree or fast approximation;
- no automatic bandwidth selection;
- no explicit reconstruction of curve connectivity.

For larger datasets, natural extensions include nearest-neighbor truncation, more aggressive batching, adaptive KDEs, and graph-based ridge reconstruction.

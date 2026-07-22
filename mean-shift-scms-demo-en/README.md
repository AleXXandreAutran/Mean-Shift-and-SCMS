# Mean Shift vs. SCMS

Implementation of **Mean Shift** and **SCMS** (*Subspace Constrained Mean Shift*) on simple 2D and 3D geometric datasets.

* **2D example:** noisy sinusoidal curve
* **3D example:** noisy helix
* **Mean Shift:** converges toward density modes
* **SCMS:** estimates density ridges

For a point $x$, the Mean Shift vector is

$$
m(x)=\frac{\sum_i w_i(x)X_i}{\sum_i w_i(x)}-x.
$$

Mean Shift applies the full displacement:

$$
x_{t+1}=x_t+m(x_t).
$$

SCMS projects this displacement onto the normal subspace of the target ridge:

$$
x_{t+1}=x_t+V(x_t)V(x_t)^\top m(x_t).
$$

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Run the examples

```bash
python examples/run_all.py
```

Generated figures are saved in `outputs/`.

## Run the tests

```bash
pytest
```

## Main parameters

* `bandwidth`: controls KDE smoothing
* `ridge_dim`: intrinsic dimension of the target ridge
* `curvature_only`: keeps points satisfying the ridge curvature condition

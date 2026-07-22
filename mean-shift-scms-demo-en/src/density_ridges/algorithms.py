"""Mean Shift and SCMS for an isotropic Gaussian KDE.

All observations and query points are represented as arrays of shape
 (n_samples, ambient_dimension).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KDEStats:
    """KDE quantities evaluated at a collection of query points."""

    density: FloatArray
    mean_shift: FloatArray
    gradient: FloatArray
    hessian: FloatArray


@dataclass(frozen=True)
class IterationResult:
    """Result returned by Mean Shift and SCMS."""

    points: FloatArray
    converged: NDArray[np.bool_]
    iterations: NDArray[np.int64]
    residual_norm: FloatArray
    density: FloatArray
    curvature_ok: NDArray[np.bool_] | None = None
    normal_eigenvalue: FloatArray | None = None


def _as_2d_float(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.ascontiguousarray(array, dtype=float)


def gaussian_kde_stats(
    data: ArrayLike,
    points: ArrayLike,
    bandwidth: float,
    *,
    chunk_size: int = 256,
) -> KDEStats:
    """Evaluate an isotropic Gaussian KDE and its first two derivatives.

    Parameters
    ----------
    data:
        Observations of shape  (n_samples, dimension).
    points:
        Query points of shape  (n_queries, dimension).
    bandwidth:
        Positive scalar bandwidth  h.
    chunk_size:
        Number of query points processed simultaneously.

    Returns
    -------
    KDEStats
        Density, Mean Shift vector, gradient and Hessian at every query point.

    Remarks 
    -----
    For the Gaussian KDE,

     gradient(p) = p * mean_shift / h**2.

    The Hessian is evaluated analytically and is used by SCMS to identify
    local normal directions.
    """

    x = _as_2d_float("data", data)
    q = _as_2d_float("points", points)

    if x.shape[1] != q.shape[1]:
        raise ValueError(
            "data and points must have the same ambient dimension; "
            f"got {x.shape[1]} and {q.shape[1]}."
        )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be a positive finite scalar.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    n_samples, dimension = x.shape
    n_queries = q.shape[0]
    h2 = bandwidth**2
    h4 = h2**2

    density = np.empty(n_queries, dtype=float)
    mean_shift_vector = np.empty((n_queries, dimension), dtype=float)
    gradient = np.empty((n_queries, dimension), dtype=float)
    hessian = np.empty((n_queries, dimension, dimension), dtype=float)

    normalizer = (2.0 * np.pi * h2) ** (-0.5 * dimension) / n_samples
    identity = np.eye(dimension)

    for start in range(0, n_queries, chunk_size):
        stop = min(start + chunk_size, n_queries)
        current = q[start:stop]

        # diff[m, n, d] = X_n - q_m
        diff = x[None, :, :] - current[:, None, :]
        squared_distance = np.einsum("mnd,mnd->mn", diff, diff)
        weights = np.exp(-0.5 * squared_distance / h2)
        weight_sum = weights.sum(axis=1)

        # In floating-point arithmetic, extremely remote points can underflow.
        safe_weight_sum = np.maximum(weight_sum, np.finfo(float).tiny)
        weighted_mean = np.einsum("mn,nd->md", weights, x) / safe_weight_sum[:, None]
        shift = weighted_mean - current

        p = normalizer * weight_sum
        outer_sum = np.einsum("mn,mni,mnj->mij", weights, diff, diff)
        hess = normalizer * (
            outer_sum / h4 - weight_sum[:, None, None] * identity / h2
        )

        density[start:stop] = p
        mean_shift_vector[start:stop] = shift
        gradient[start:stop] = p[:, None] * shift / h2
        hessian[start:stop] = hess

    return KDEStats(
        density=density,
        mean_shift=mean_shift_vector,
        gradient=gradient,
        hessian=hessian,
    )


def _clip_steps(steps: FloatArray, max_step: float | None) -> FloatArray:
    if max_step is None:
        return steps
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be None or a positive finite scalar.")

    norms = np.linalg.norm(steps, axis=1)
    scales = np.minimum(1.0, max_step / np.maximum(norms, np.finfo(float).tiny))
    return steps * scales[:, None]


def mean_shift(
    data: ArrayLike,
    seeds: ArrayLike,
    bandwidth: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    step_size: float = 1.0,
    max_step: float | None = None,
    chunk_size: int = 256,
) -> IterationResult:
    """Run Gaussian Mean Shift from multiple initial seeds."""

    x = _as_2d_float("data", data)
    current = _as_2d_float("seeds", seeds).copy()

    if x.shape[1] != current.shape[1]:
        raise ValueError("data and seeds must have the same ambient dimension.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("tol must be positive and finite.")
    if step_size <= 0 or not np.isfinite(step_size):
        raise ValueError("step_size must be positive and finite.")

    n_seeds = current.shape[0]
    converged = np.zeros(n_seeds, dtype=bool)
    iterations = np.zeros(n_seeds, dtype=np.int64)
    residual = np.full(n_seeds, np.inf, dtype=float)

    for iteration in range(1, max_iter + 1):
        active = ~converged
        if not np.any(active):
            break

        stats = gaussian_kde_stats(
            x, current[active], bandwidth, chunk_size=chunk_size
        )
        active_indices = np.flatnonzero(active)
        residual_active = np.linalg.norm(stats.mean_shift, axis=1)
        residual[active_indices] = residual_active

        newly_converged = residual_active <= tol
        if np.any(newly_converged):
            indices = active_indices[newly_converged]
            converged[indices] = True
            iterations[indices] = iteration

        moving_indices = active_indices[~newly_converged]
        if moving_indices.size:
            steps = step_size * stats.mean_shift[~newly_converged]
            steps = _clip_steps(steps, max_step)
            current[moving_indices] += steps

    iterations[~converged] = max_iter
    final_stats = gaussian_kde_stats(x, current, bandwidth, chunk_size=chunk_size)
    residual = np.linalg.norm(final_stats.mean_shift, axis=1)
    converged = residual <= tol

    return IterationResult(
        points=current,
        converged=converged,
        iterations=iterations,
        residual_norm=residual,
        density=final_stats.density,
    )


def _normal_subspace(
    hessian: FloatArray,
    ridge_dim: int,
) -> tuple[FloatArray, FloatArray]:
    """Return normal eigenvectors and the ridge curvature eigenvalue.

     numpy.linalg.eigh  returns eigenvalues in ascending order. For a
     ridge_dim -dimensional ridge in ambient dimension  D , the normal
    subspace is spanned by the  D - ridge_dim  smallest-eigenvalue vectors.
    """

    dimension = hessian.shape[-1]
    if not 0 <= ridge_dim < dimension:
        raise ValueError(
            f"ridge_dim must satisfy 0 <= ridge_dim < {dimension}; "
            f"got {ridge_dim}."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    normal_count = dimension - ridge_dim
    normal_vectors = eigenvectors[:, :, :normal_count]

    # Largest eigenvalue among the normal directions. In the common descending
    # notation lambda_1 >= ... >= lambda_D, this is lambda_{ridge_dim + 1}.
    normal_curvature = eigenvalues[:, normal_count - 1]
    return normal_vectors, normal_curvature


def scms(
    data: ArrayLike,
    seeds: ArrayLike,
    bandwidth: float,
    *,
    ridge_dim: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    step_size: float = 1.0,
    max_step: float | None = None,
    curvature_only: bool = True,
    curvature_tol: float = 0.0,
    chunk_size: int = 256,
) -> IterationResult:
    """Run Subspace Constrained Mean Shift.

    At each iteration, the Mean Shift vector is projected onto the local
    normal subspace estimated from the KDE Hessian.

    Parameters
    ----------
    ridge_dim:
        Intrinsic dimension of the target ridge. Use 1 for a curve and 2 for
        a surface embedded in 3D.
    curvature_only:
        If true, convergence additionally requires negative normal curvature.
        All points are still returned; inspect  curvature_ok  to filter them.
    curvature_tol:
        Require the ridge curvature eigenvalue to be less than
         -curvature_tol .
    """

    x = _as_2d_float("data", data)
    current = _as_2d_float("seeds", seeds).copy()

    if x.shape[1] != current.shape[1]:
        raise ValueError("data and seeds must have the same ambient dimension.")
    if not 0 <= ridge_dim < x.shape[1]:
        raise ValueError(
            f"ridge_dim must satisfy 0 <= ridge_dim < {x.shape[1]}."
        )
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("tol must be positive and finite.")
    if step_size <= 0 or not np.isfinite(step_size):
        raise ValueError("step_size must be positive and finite.")
    if curvature_tol < 0 or not np.isfinite(curvature_tol):
        raise ValueError("curvature_tol must be non-negative and finite.")

    n_seeds = current.shape[0]
    converged = np.zeros(n_seeds, dtype=bool)
    iterations = np.zeros(n_seeds, dtype=np.int64)
    residual = np.full(n_seeds, np.inf, dtype=float)

    for iteration in range(1, max_iter + 1):
        active = ~converged
        if not np.any(active):
            break

        active_indices = np.flatnonzero(active)
        stats = gaussian_kde_stats(
            x, current[active], bandwidth, chunk_size=chunk_size
        )
        normal_vectors, normal_curvature = _normal_subspace(
            stats.hessian, ridge_dim
        )

        # Projection V V^T m without explicitly forming every projector.
        normal_coordinates = np.einsum(
            "mdi,md->mi", normal_vectors, stats.mean_shift
        )
        projected_shift = np.einsum(
            "mdi,mi->md", normal_vectors, normal_coordinates
        )
        residual_active = np.linalg.norm(projected_shift, axis=1)
        residual[active_indices] = residual_active

        curvature_ok = normal_curvature < -curvature_tol
        stopping = residual_active <= tol
        if curvature_only:
            stopping &= curvature_ok

        if np.any(stopping):
            indices = active_indices[stopping]
            converged[indices] = True
            iterations[indices] = iteration

        moving_indices = active_indices[~stopping]
        if moving_indices.size:
            steps = step_size * projected_shift[~stopping]
            steps = _clip_steps(steps, max_step)
            current[moving_indices] += steps

    iterations[~converged] = max_iter

    final_stats = gaussian_kde_stats(x, current, bandwidth, chunk_size=chunk_size)
    final_normal_vectors, final_normal_curvature = _normal_subspace(
        final_stats.hessian, ridge_dim
    )
    final_coordinates = np.einsum(
        "mdi,md->mi", final_normal_vectors, final_stats.mean_shift
    )
    final_projected_shift = np.einsum(
        "mdi,mi->md", final_normal_vectors, final_coordinates
    )
    residual = np.linalg.norm(final_projected_shift, axis=1)
    curvature_ok = final_normal_curvature < -curvature_tol
    converged = residual <= tol
    if curvature_only:
        converged &= curvature_ok

    return IterationResult(
        points=current,
        converged=converged,
        iterations=iterations,
        residual_norm=residual,
        density=final_stats.density,
        curvature_ok=curvature_ok,
        normal_eigenvalue=final_normal_curvature,
    )

"""Synthetic geometric datasets for the examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _sample_mixture_parameter(
    rng: np.random.Generator,
    n_samples: int,
    low: float,
    high: float,
) -> FloatArray:
    """Sample a non-uniform one-dimensional parameter.

    The non-uniformity is intentional: it gives Mean Shift a tangential
    density gradient, while SCMS can still recover the underlying ridge.
    """

    component = rng.random(n_samples) < 0.65
    parameter = np.empty(n_samples, dtype=float)

    center_1 = low + 0.35 * (high - low)
    center_2 = low + 0.75 * (high - low)
    scale_1 = 0.16 * (high - low)
    scale_2 = 0.10 * (high - low)

    parameter[component] = rng.normal(center_1, scale_1, component.sum())
    parameter[~component] = rng.normal(center_2, scale_2, (~component).sum())
    return np.clip(parameter, low, high)


def make_sine_curve_2d(
    n_samples: int = 700,
    *,
    noise: float = 0.12,
    random_state: int = 7,
) -> tuple[FloatArray, FloatArray]:
    """Return noisy observations and the noiseless 2D reference curve."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(random_state)
    t = _sample_mixture_parameter(rng, n_samples, -3.2, 3.2)

    x = t
    y = 0.65 * np.sin(1.35 * t)
    clean_samples = np.column_stack([x, y])

    # Add approximately normal noise to the curve rather than only vertical
    # noise, which makes the geometric ridge interpretation clearer.
    tangent = np.column_stack(
        [
            np.ones_like(t),
            0.65 * 1.35 * np.cos(1.35 * t),
        ]
    )
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    noisy = clean_samples + rng.normal(0.0, noise, (n_samples, 1)) * normal

    reference_t = np.linspace(-3.2, 3.2, 500)
    reference = np.column_stack(
        [reference_t, 0.65 * np.sin(1.35 * reference_t)]
    )
    return noisy, reference


def make_helix_3d(
    n_samples: int = 900,
    *,
    noise: float = 0.10,
    random_state: int = 11,
) -> tuple[FloatArray, FloatArray]:
    """Return noisy observations and a noiseless 3D helix."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(random_state)
    t = _sample_mixture_parameter(rng, n_samples, 0.0, 4.0 * np.pi)

    clean_samples = np.column_stack(
        [
            np.cos(t),
            np.sin(t),
            0.16 * t - 1.0,
        ]
    )

    tangent = np.column_stack(
        [
            -np.sin(t),
            np.cos(t),
            np.full_like(t, 0.16),
        ]
    )
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    # Build two orthonormal normal vectors at each helix point.
    reference_axis = np.tile(np.array([0.0, 0.0, 1.0]), (n_samples, 1))
    normal_1 = np.cross(tangent, reference_axis)
    normal_1_norm = np.linalg.norm(normal_1, axis=1, keepdims=True)

    # The helix tangent is never parallel to the z-axis, but keep a safe
    # fallback for numerical robustness.
    fallback = normal_1_norm[:, 0] < 1e-12
    if np.any(fallback):
        reference_axis[fallback] = np.array([1.0, 0.0, 0.0])
        normal_1[fallback] = np.cross(
            tangent[fallback], reference_axis[fallback]
        )
        normal_1_norm[fallback] = np.linalg.norm(
            normal_1[fallback], axis=1, keepdims=True
        )

    normal_1 /= normal_1_norm
    normal_2 = np.cross(tangent, normal_1)

    coefficients = rng.normal(0.0, noise, (n_samples, 2))
    noisy = (
        clean_samples
        + coefficients[:, [0]] * normal_1
        + coefficients[:, [1]] * normal_2
    )

    reference_t = np.linspace(0.0, 4.0 * np.pi, 700)
    reference = np.column_stack(
        [
            np.cos(reference_t),
            np.sin(reference_t),
            0.16 * reference_t - 1.0,
        ]
    )
    return noisy, reference

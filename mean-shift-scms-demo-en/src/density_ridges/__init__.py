"""Educational Mean Shift and SCMS implementation."""

from .algorithms import (
    IterationResult,
    gaussian_kde_stats,
    mean_shift,
    scms,
)
from .synthetic import make_helix_3d, make_sine_curve_2d

__all__ = [
    "IterationResult",
    "gaussian_kde_stats",
    "mean_shift",
    "scms",
    "make_helix_3d",
    "make_sine_curve_2d",
]

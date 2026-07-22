"""Compare Mean Shift and SCMS on a noisy 2D sine curve."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from density_ridges import make_sine_curve_2d, mean_shift, scms


DATA_COLOR = "#222222"
REFERENCE_COLOR = "#0066FF"
SEED_COLOR = "#FF8C00"
MEAN_SHIFT_COLOR = "#E60026"
SCMS_COLOR = "#00A651"


def _save_data_plot(data: np.ndarray, reference: np.ndarray, output: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=14,
        alpha=0.42,
        c=DATA_COLOR,
        edgecolors="none",
        label="Noisy observations",
    )
    plt.plot(
        reference[:, 0],
        reference[:, 1],
        linewidth=3.0,
        c=REFERENCE_COLOR,
        label="True curve",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic 2D data")
    plt.grid(alpha=0.18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _save_result_plot(
    data: np.ndarray,
    reference: np.ndarray,
    seeds: np.ndarray,
    result: np.ndarray,
    output: Path,
    title: str,
    result_color: str,
    result_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=11,
        alpha=0.22,
        c=DATA_COLOR,
        edgecolors="none",
        label="Noisy observations",
    )
    plt.plot(
        reference[:, 0],
        reference[:, 1],
        linewidth=3.0,
        c=REFERENCE_COLOR,
        label="True curve",
    )
    plt.scatter(
        seeds[:, 0],
        seeds[:, 1],
        s=20,
        alpha=0.48,
        c=SEED_COLOR,
        edgecolors="none",
        label="Initial seeds",
    )
    plt.scatter(
        result[:, 0],
        result[:, 1],
        s=30,
        alpha=0.92,
        c=result_color,
        edgecolors="white",
        linewidths=0.35,
        label=result_label,
        zorder=5,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(alpha=0.18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    data, reference = make_sine_curve_2d()
    rng = np.random.default_rng(123)
    seeds = data[rng.choice(len(data), size=260, replace=False)].copy()
    seeds += rng.normal(0.0, 0.05, seeds.shape)

    bandwidth = 0.34

    mean_shift_result = mean_shift(
        data,
        seeds,
        bandwidth,
        max_iter=120,
        tol=2e-4,
        max_step=0.6 * bandwidth,
    )
    scms_result = scms(
        data,
        seeds,
        bandwidth,
        ridge_dim=1,
        max_iter=160,
        tol=2e-4,
        max_step=0.6 * bandwidth,
        curvature_only=True,
    )

    scms_points = scms_result.points[scms_result.converged]

    _save_data_plot(data, reference, output_dir / "2d_data.png")
    _save_result_plot(
        data,
        reference,
        seeds,
        mean_shift_result.points,
        output_dir / "2d_mean_shift.png",
        "Mean Shift in 2D: convergence toward density modes",
        MEAN_SHIFT_COLOR,
        "Mean Shift endpoints",
    )
    _save_result_plot(
        data,
        reference,
        seeds,
        scms_points,
        output_dir / "2d_scms.png",
        "SCMS in 2D: estimation of a one-dimensional density ridge",
        SCMS_COLOR,
        "SCMS ridge estimate",
    )

    print(
        f"2D Mean Shift: {mean_shift_result.converged.sum()}/{len(seeds)} "
        "seeds converged."
    )
    print(
        f"2D SCMS: {scms_result.converged.sum()}/{len(seeds)} "
        "seeds satisfy convergence and curvature conditions."
    )
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()

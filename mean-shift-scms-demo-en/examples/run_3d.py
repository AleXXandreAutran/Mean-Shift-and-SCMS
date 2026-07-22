"""Compare Mean Shift and SCMS on a noisy 3D helix."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from density_ridges import make_helix_3d, mean_shift, scms


DATA_COLOR = "#222222"
REFERENCE_COLOR = "#0066FF"
SEED_COLOR = "#FF8C00"
MEAN_SHIFT_COLOR = "#E60026"
SCMS_COLOR = "#00A651"


def _format_axes(axis) -> None:
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.view_init(elev=24, azim=-58)
    axis.grid(True, alpha=0.20)


def _save_data_plot(data: np.ndarray, reference: np.ndarray, output: Path) -> None:
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        s=10,
        alpha=0.34,
        c=DATA_COLOR,
        edgecolors="none",
        label="Noisy observations",
    )
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        linewidth=3.2,
        c=REFERENCE_COLOR,
        label="True helix",
    )
    axis.set_title("Synthetic 3D data")
    _format_axes(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


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
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        s=8,
        alpha=0.16,
        c=DATA_COLOR,
        edgecolors="none",
        label="Noisy observations",
    )
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        linewidth=3.2,
        c=REFERENCE_COLOR,
        label="True helix",
    )
    axis.scatter(
        seeds[:, 0],
        seeds[:, 1],
        seeds[:, 2],
        s=16,
        alpha=0.34,
        c=SEED_COLOR,
        edgecolors="none",
        label="Initial seeds",
    )
    axis.scatter(
        result[:, 0],
        result[:, 1],
        result[:, 2],
        s=24,
        alpha=0.94,
        c=result_color,
        edgecolors="white",
        linewidths=0.30,
        label=result_label,
    )
    axis.set_title(title)
    _format_axes(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    data, reference = make_helix_3d()
    rng = np.random.default_rng(321)
    seeds = data[rng.choice(len(data), size=320, replace=False)].copy()
    seeds += rng.normal(0.0, 0.04, seeds.shape)

    bandwidth = 0.28

    mean_shift_result = mean_shift(
        data,
        seeds,
        bandwidth,
        max_iter=140,
        tol=3e-4,
        max_step=0.55 * bandwidth,
    )
    scms_result = scms(
        data,
        seeds,
        bandwidth,
        ridge_dim=1,
        max_iter=180,
        tol=3e-4,
        max_step=0.55 * bandwidth,
        curvature_only=True,
    )

    scms_points = scms_result.points[scms_result.converged]

    _save_data_plot(data, reference, output_dir / "3d_data.png")
    _save_result_plot(
        data,
        reference,
        seeds,
        mean_shift_result.points,
        output_dir / "3d_mean_shift.png",
        "Mean Shift in 3D: concentration toward density modes",
        MEAN_SHIFT_COLOR,
        "Mean Shift endpoints",
    )
    _save_result_plot(
        data,
        reference,
        seeds,
        scms_points,
        output_dir / "3d_scms.png",
        "SCMS in 3D: estimation of a helical density ridge",
        SCMS_COLOR,
        "SCMS ridge estimate",
    )

    print(
        f"3D Mean Shift: {mean_shift_result.converged.sum()}/{len(seeds)} "
        "seeds converged."
    )
    print(
        f"3D SCMS: {scms_result.converged.sum()}/{len(seeds)} "
        "seeds satisfy convergence and curvature conditions."
    )
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()

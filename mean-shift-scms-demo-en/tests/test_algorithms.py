import numpy as np

from density_ridges import gaussian_kde_stats, mean_shift, scms


def test_kde_derivative_shapes() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3))
    points = rng.normal(size=(7, 3))

    stats = gaussian_kde_stats(data, points, bandwidth=0.8)

    assert stats.density.shape == (7,)
    assert stats.mean_shift.shape == (7, 3)
    assert stats.gradient.shape == (7, 3)
    assert stats.hessian.shape == (7, 3, 3)
    assert np.all(np.isfinite(stats.hessian))


def test_mean_shift_moves_toward_single_cluster_center() -> None:
    rng = np.random.default_rng(1)
    center = np.array([1.5, -0.7])
    data = center + rng.normal(scale=0.15, size=(250, 2))
    seeds = np.array([[-1.0, 1.0], [3.0, -2.0], [0.0, -0.5]])

    result = mean_shift(
        data,
        seeds,
        bandwidth=0.5,
        max_iter=100,
        tol=1e-5,
    )

    distances = np.linalg.norm(result.points - center, axis=1)
    assert np.all(distances < 0.12)


def test_scms_with_ridge_dim_zero_matches_mean_shift() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=(120, 2))
    seeds = rng.normal(size=(15, 2))

    ms_result = mean_shift(
        data,
        seeds,
        bandwidth=0.9,
        max_iter=8,
        tol=1e-14,
    )
    scms_result = scms(
        data,
        seeds,
        bandwidth=0.9,
        ridge_dim=0,
        max_iter=8,
        tol=1e-14,
        curvature_only=False,
    )

    np.testing.assert_allclose(scms_result.points, ms_result.points, atol=1e-10)


def test_scms_recovers_noisy_horizontal_line() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-2.5, 2.5, size=500)
    y = rng.normal(0.0, 0.10, size=500)
    data = np.column_stack([x, y])

    seed_x = np.linspace(-2.0, 2.0, 40)
    seeds = np.column_stack([seed_x, np.full_like(seed_x, 0.25)])

    result = scms(
        data,
        seeds,
        bandwidth=0.35,
        ridge_dim=1,
        max_iter=120,
        tol=2e-4,
        curvature_only=True,
    )

    accepted = result.points[result.converged]
    assert len(accepted) >= 30
    assert np.median(np.abs(accepted[:, 1])) < 0.08

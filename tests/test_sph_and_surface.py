import numpy as np

from discsim import DiscSurface, SPHGas, SPHSettings, thin_disc
from discsim.sph import kernel_w


def test_kernel_normalised():
    r = np.linspace(0, 2, 4001)
    integral = np.trapezoid(4 * np.pi * r**2 * kernel_w(r, np.ones_like(r)), r)
    assert np.isclose(integral, 1.0, rtol=1e-3)


def test_sph_forces_conserve_momentum():
    pos, vel, m = thin_disc(1500, 1.0, 3.0, aspect_ratio=0.05, seed=2)
    gas = SPHGas(SPHSettings(aspect_ratio=0.05))
    a = gas.acceleration(pos, vel + np.random.default_rng(0).normal(0, 0.01, vel.shape), m)
    assert np.allclose((m[:, None] * a).sum(0), 0.0, atol=1e-10 * np.abs(m[:, None] * a).sum())


def test_sph_vertical_pressure_balances_gravity_in_equilibrium_disc():
    # in the initial Gaussian disc, the vertical pressure acceleration should
    # roughly cancel the vertical component of gravity, -z / R^3
    pos, vel, m = thin_disc(20000, 1.0, 3.0, aspect_ratio=0.1, seed=4)
    gas = SPHGas(SPHSettings(aspect_ratio=0.1, n_neighbours=50))
    a = gas.acceleration(pos, vel, m)
    R = np.hypot(pos[:, 0], pos[:, 1])
    sel = (R > 1.6) & (R < 2.4) & (np.abs(pos[:, 2]) > 0.05 * R) & (np.abs(pos[:, 2]) < 0.15 * R)
    ratio = np.median(a[sel, 2] / (pos[sel, 2] / R[sel] ** 3))
    assert 0.5 < ratio < 1.5


def test_surface_from_particles_recovers_gaussian_width():
    pos, _, _ = thin_disc(200000, 1.0, 5.0, aspect_ratio=0.05, gas=True, seed=5)
    s = DiscSurface.from_particles(pos, 1.0, 5.0, n_r=10, n_phi=8, n_sigma=1.0)
    R, _ = s.mesh
    assert np.allclose(s.z / R, 0.05, rtol=0.15)


def test_photosphere_holes_where_gas_is_missing():
    pos, _, m = thin_disc(100000, 1.0, 5.0, aspect_ratio=0.05, seed=6)
    R = np.hypot(pos[:, 0], pos[:, 1])
    keep = (R < 2.5) | (R > 3.0)           # carve a gap
    mass_g = np.full(keep.sum(), 1e33)
    s = DiscSurface.from_particles(pos[keep], 1.0, 5.0, n_r=40, n_phi=8, kappa=0.4, mass_g=mass_g, min_count=3)
    gap = (s.r > 2.55) & (s.r < 2.95)
    assert not s.valid[gap].any()
    assert s.valid[(s.r > 1.2) & (s.r < 2.3)].all()

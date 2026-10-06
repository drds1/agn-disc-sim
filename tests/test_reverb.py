import numpy as np

from discsim import DiscPhysics, DiscSurface, Observer, irradiate, response_functions, starkey2023, thin_disc_lag
from discsim.reverb import shadow_mask

PHYS = DiscPhysics(m_bh=1e8, mdot=0.5, l_lamppost=1e44, h_lamppost_rg=10.0)
WAV = np.array([2500.0, 4000.0, 6000.0, 9000.0])


def _flat(r_out=30.0, **kw):
    return irradiate(DiscSurface.flat(PHYS.r_in_ld, r_out, n_r=400, n_phi=90, **kw), PHYS)


def test_responses_are_causal_and_normalised():
    rs = response_functions(_flat(), Observer(45.0), WAV, tau_max=60.0, n_tau=800)
    assert np.all(rs.psi >= 0)
    assert np.allclose(np.trapezoid(rs.psi, rs.tau, axis=1), 1.0)
    assert np.all(rs.tau > 0)


def test_flat_disc_lag_scales_as_lambda_four_thirds():
    rs = response_functions(_flat(), Observer(0.0), WAV, tau_max=60.0, n_tau=1500)
    assert abs(rs.lag_slope() - 4 / 3) < 0.08


def test_flat_disc_mean_lag_matches_analytic_response_weighted_radius():
    rs = response_functions(_flat(), Observer(0.0), WAV, tau_max=60.0, n_tau=1500)
    analytic = thin_disc_lag(WAV, PHYS.m_bh, PHYS.mdot, PHYS.l_lamppost, 0.0, PHYS.h_lp_ld, x=3.36)
    assert np.allclose(rs.mean_lag(), analytic, rtol=0.06)


def test_inclination_does_not_change_the_mean_lag_of_a_flat_disc():
    disc = _flat()
    a = response_functions(disc, Observer(0.0), WAV, tau_max=80.0, n_tau=1500).mean_lag()
    b = response_functions(disc, Observer(60.0), WAV, tau_max=80.0, n_tau=1500).mean_lag()
    # only the lamppost-height term h cos(i) of the delay depends on inclination
    h = PHYS.h_lp_ld
    assert np.allclose(a - h, b - h * np.cos(np.radians(60.0)), rtol=0.005)


def test_rim_lengthens_optical_lags_but_not_far_uv():
    phys = DiscPhysics(m_bh=5e7, mdot=0.05, l_lamppost=3e43)
    flat = irradiate(DiscSurface.flat(phys.r_in_ld, 5.0, n_r=400, n_phi=90), phys)
    rim = irradiate(DiscSurface.from_function(starkey2023(0.15, 5.0, 100.0), phys.r_in_ld, 5.0, n_r=400,
                                              n_phi=90), phys)
    obs, wav = Observer(30.0), np.array([1500.0, 8000.0])
    lf = response_functions(flat, obs, wav, tau_max=15.0).mean_lag()
    lr = response_functions(rim, obs, wav, tau_max=15.0).mean_lag()
    assert lr[1] / lf[1] > 1.5
    assert lr[0] / lf[0] < 1.15


def test_shadow_behind_a_wall():
    def wall(R, P):
        return np.where((R > 2.0) & (R < 2.2), 0.5, 0.0)
    surf = DiscSurface.from_function(wall, 0.1, 6.0, n_r=300, n_phi=8)
    lit = shadow_mask(surf, h_lp=0.05)
    r = surf.r
    face = np.argmax(r > 2.0)
    assert lit[face].all()                           # the wall's inner face is lit
    assert not lit[face + 1:][r[face + 1:] < 6.0].any()  # its flat top and everything behind it are dark
    assert lit[r < 1.9].all()

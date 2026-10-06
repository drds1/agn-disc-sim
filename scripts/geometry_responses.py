"""
How the disc's shape changes its delays and its brightness: flat, flared,
rimmed (Starkey et al. 2023) and spiral-corrugated surfaces, all with the same
black hole, accretion rate and lamppost.

The point of the comparison is the disc-size problem's two halves: a geometry
that lengthens the delays is only a solution if it does so without making the
disc correspondingly brighter (the Fairall 9 / NGC 5548 "equivalent H0" test in
echofit). So the figure shows lag spectra *and* each geometry's SED.

    python scripts/geometry_responses.py [--out outputs/geometry]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discsim import DiscPhysics, DiscSurface, Observer, flared, irradiate, response_functions, sed, spiral, starkey2023
from discsim.reverb import thin_disc_lag

M_BH = 5.0e7         # NGC 5548-like
MDOT = 0.05          # Msun / yr
L_LP = 3.0e43        # erg / s
R_OUT = 5.0          # light-days, as in Starkey et al. 2023's rim
WAV = np.geomspace(1200.0, 9500.0, 14)


def geometries():
    flat = lambda R, P: 0.0 * R
    return {
        "flat": flat,
        "flared H/R=0.01 (R/1ld)^(1/8)": flared(0.01, 1.0, 9.0 / 8.0),
        "rim (Starkey+23)": starkey2023(height=0.15, r_ref=R_OUT, beta=100.0),
        "ripples (Starkey+23)": starkey2023(height=0.03, r_ref=R_OUT, beta=1.0, ripple_amplitude=0.6,
                                             ripple_wavelength=1.0, ripple_slope=-3.0),
        "2-arm spiral (5% of R)": spiral(0.05, m=2, pitch_deg=20.0, base=flared(0.01, 1.0, 9.0 / 8.0)),
    }


def main(out: Path, inclination: float = 30.0):
    out.mkdir(parents=True, exist_ok=True)
    phys = DiscPhysics(m_bh=M_BH, mdot=MDOT, l_lamppost=L_LP, h_lamppost_rg=10.0)
    obs = Observer(inclination, distance_mpc=75.0)
    print(f"Eddington ratio {phys.eddington_ratio:.3f}, lamppost height {phys.h_lp_ld:.4f} ld")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    rows = []
    for name, h in geometries().items():
        surf = DiscSurface.from_function(h, phys.r_in_ld, R_OUT, n_r=500, n_phi=180)
        disc = irradiate(surf, phys)
        rs = response_functions(disc, obs, WAV, tau_max=15.0, n_tau=600)
        f = sed(disc, obs, WAV)
        lag = rs.mean_lag()
        rows.append((name, lag, f, disc.lit.mean()))
        ax[0, 0].plot(surf.r, surf.z[:, 0] / surf.r, label=name)
        ax[0, 1].plot(rs.tau, rs.psi[np.argmin(abs(WAV - 5100))], label=name)
        ax[1, 0].plot(WAV, lag, "o-", ms=3, label=name)
        ax[1, 1].plot(WAV, f, label=name)
        print(f"{name:32s} lit {disc.lit.mean():5.2f}  <tau>(5100) {np.interp(5100, WAV, lag):5.2f} d  "
              f"slope {rs.lag_slope():4.2f}  f_nu(5100) {np.interp(5100, WAV, f):6.2f} mJy")

    ax[1, 0].plot(WAV, thin_disc_lag(WAV, M_BH, MDOT, L_LP, 0.0, phys.h_lp_ld, x=3.36), "k--",
                  label="analytic thin disc (X=3.36)")
    ax[0, 0].set(xscale="log", xlabel="R (light-days)", ylabel="H/R at azimuth 0", title="surface")
    ax[0, 1].set(xlabel="delay (days)", ylabel="psi(tau), 5100 A", title=f"response, i = {inclination:.0f} deg",
                 xlim=(0, 8))
    ax[1, 0].set(xscale="log", yscale="log", xlabel="wavelength (A)", ylabel="mean delay (days)", title="lag spectrum")
    ax[1, 1].set(xscale="log", yscale="log", xlabel="wavelength (A)", ylabel="f_nu (mJy)", title="disc SED at 75 Mpc")
    for a in ax.flat:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "geometry_responses.png", dpi=130)
    print("wrote", out / "geometry_responses.png")

    flat_lag, flat_f = rows[0][1], rows[0][2]
    print("\nlag boost vs flux boost at 5100 A (what the disc-size problem needs: lag x3, flux x1)")
    for name, lag, f, _ in rows:
        print(f"  {name:32s} lag x{np.interp(5100, WAV, lag / flat_lag):4.2f}   flux x{np.interp(5100, WAV, f / flat_f):4.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/geometry")
    p.add_argument("--inclination", type=float, default=30.0)
    a = p.parse_args()
    main(Path(a.out), a.inclination)

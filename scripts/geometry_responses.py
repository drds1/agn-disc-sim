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
        # corrugation smaller than the base height, so the surface stays above the mid-plane
        "2-arm spiral (±0.006 R)": spiral(0.006, m=2, pitch_deg=20.0, base=flared(0.01, 1.0, 9.0 / 8.0)),
    }


def side_view(ax, surf, disc, h_lp: float, colour, name: str):
    """H against R at azimuth 0, with the lamppost and the rays that set the shadows."""
    r, z, lit = surf.r, surf.z[:, 0], disc.lit[:, 0]
    ax.fill_between(r, 0, z, color="0.88", lw=0)
    ax.plot(r, np.where(lit, z, np.nan), color=colour, lw=1.8)
    ax.plot(r, z, color=colour, lw=1, alpha=0.3)
    # a shadow starts where a lit element is followed by a dark one: the ray from the lamp through that
    # element grazes it and passes over the shadowed stretch behind
    # element grazes it and passes over the shadowed stretch behind, until the surface is lit again
    edges = np.nonzero(lit[:-1] & ~lit[1:])[0]
    ends = [next((j for j in range(i + 1, len(r)) if lit[j]), len(r) - 1) for i in edges]
    rays = sorted(zip(edges, ends), key=lambda e: r[e[0]] - r[e[1]])[:3] if len(edges) else [(len(r) - 1,) * 2]
    for i, j in rays:
        slope = (z[i] - h_lp) / r[i]
        ax.plot([0, r[j]], [h_lp, h_lp + slope * r[j]], color="0.45", lw=0.8)
    ax.plot(0, h_lp, "*", color="k", ms=9, clip_on=False, zorder=5)
    ax.set(xlim=(-0.1, r[-1] * 1.02), ylim=(min(-0.004, z.min() * 1.1), max(z.max(), h_lp) * 1.25),
           xlabel="R (ld)", title=name.split(" (")[0].split(" H/R")[0], )
    ax.title.set_fontsize(9)
    ax.tick_params(labelsize=7)


def main(out: Path, inclination: float = 30.0):
    out.mkdir(parents=True, exist_ok=True)
    phys = DiscPhysics(m_bh=M_BH, mdot=MDOT, l_lamppost=L_LP, h_lamppost_rg=10.0)
    obs = Observer(inclination, distance_mpc=75.0)
    print(f"Eddington ratio {phys.eddington_ratio:.3f}, lamppost height {phys.h_lp_ld:.4f} ld")

    fig = plt.figure(figsize=(11, 11.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.6, 1])
    ax = top.subplots(2, 2)
    side = bottom.subplots(1, len(geometries()))
    rows = []
    for k, (name, h) in enumerate(geometries().items()):
        surf = DiscSurface.from_function(h, phys.r_in_ld, R_OUT, n_r=500, n_phi=180)
        disc = irradiate(surf, phys)
        rs = response_functions(disc, obs, WAV, tau_max=15.0, n_tau=600)
        f = sed(disc, obs, WAV)
        lag = rs.mean_lag()
        rows.append((name, lag, f, disc.lit.mean()))
        # solid where the lamppost lights the surface, faded where it is shadowed
        hr = surf.z[:, 0] / surf.r
        line, = ax[0, 0].plot(surf.r, np.where(disc.lit[:, 0], hr, np.nan), label=f"{name} (lit {disc.lit.mean():.0%})")
        ax[0, 0].plot(surf.r, hr, color=line.get_color(), alpha=0.25, lw=1)
        ax[0, 1].plot(rs.tau, rs.psi[np.argmin(abs(WAV - 5100))], label=name)
        side_view(side[k], surf, disc, phys.h_lp_ld, line.get_color(), name)
        ax[1, 0].plot(WAV, lag, "o-", ms=3, label=name)
        ax[1, 1].plot(WAV, f, label=name)
        print(f"{name:32s} lit {disc.lit.mean():5.2f}  <tau>(5100) {np.interp(5100, WAV, lag):5.2f} d  "
              f"slope {rs.lag_slope():4.2f}  f_nu(5100) {np.interp(5100, WAV, f):6.2f} mJy")

    ax[1, 0].plot(WAV, thin_disc_lag(WAV, M_BH, MDOT, L_LP, 0.0, phys.h_lp_ld, x=3.36), "k--",
                  label="analytic thin disc (X=3.36)")
    r = np.geomspace(phys.r_in_ld, R_OUT, 200)
    ax[0, 0].plot(r, phys.h_lp_ld / r, "k:", lw=1.2, label=f"lamppost height / R ({phys.h_lamppost_rg:g} r$_g$)")
    ax[0, 0].set(xscale="log", xlabel="R (light-days)", ylabel="H/R at azimuth 0", ylim=(-0.002, 0.04),
                 title="surface (faded: in shadow)")
    ax[0, 1].set(xlabel="delay (days)", ylabel="psi(tau), 5100 A", title=f"response, i = {inclination:.0f} deg",
                 xlim=(0, 8))
    ax[1, 0].set(xscale="log", yscale="log", xlabel="wavelength (A)", ylabel="mean delay (days)", title="lag spectrum")
    ax[1, 1].set(xscale="log", yscale="log", xlabel="wavelength (A)", ylabel="f_nu (mJy)", title="disc SED at 75 Mpc")
    for a in ax.flat:
        a.legend(fontsize=7)
    side[0].set_ylabel("H (light-days)")
    bottom.suptitle("Side view in true height on linear axes, so the lamppost's rays are straight lines "
                    "(vertical scale differs per panel)\nstar: lamppost;  grey: rays that graze the surface;  "
                    "faded: in shadow", fontsize=9)
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

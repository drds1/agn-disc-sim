"""
Simulated disc -> mock multi-band campaign in pycream2's input format.

Builds the rimmed disc of Starkey et al. (2023) (or a flat one), drives it
with a damped random walk, samples it like a Swift + ground campaign, and
writes one text file per band plus the pycream2 command that fits them with
the standard flat thin disc. Comparing the fitted log_mdot / inclination with
the truth measures how badly a non-flat disc biases a CREAM-style fit, and
whether the bias looks like the disc-size problem.

    python scripts/mock_campaign.py [--geometry rim|flat] [--out outputs/mock_rim]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discsim import (DiscPhysics, DiscSurface, MockCampaign, Observer, drw_driver, echo, irradiate,
                     response_functions, sed, starkey2023, write_pycream2)
from discsim.lightcurves import pycream2_command

BANDS = {"uvw2": 1928.0, "uvm2": 2246.0, "uvw1": 2600.0, "u": 3465.0, "g": 4770.0, "r": 6215.0,
         "i": 7545.0, "z": 8700.0}
M_BH = 5.0e7


def main(a):
    out = Path(a.out)
    phys = DiscPhysics(m_bh=M_BH, mdot=0.05, l_lamppost=3e43, h_lamppost_rg=10.0)
    obs = Observer(a.inclination, 75.0)
    h = starkey2023(height=0.15, r_ref=5.0, beta=100.0) if a.geometry == "rim" else (lambda R, P: 0 * R)
    disc = irradiate(DiscSurface.from_function(h, phys.r_in_ld, 5.0, n_r=500, n_phi=180), phys)
    wav = np.array(list(BANDS.values()))
    rs = response_functions(disc, obs, wav, tau_max=15.0, n_tau=600)
    mean_flux = sed(disc, obs, wav)

    t, x = drw_driver(a.days + 40.0, dt=0.05, tau_drw=60.0, sigma=0.15, seed=a.seed)
    curves = echo(t, x, rs)
    # fractional variability of the reprocessed light scales the steady flux;
    # 30% of each band's flux is made to vary, the rest is steady (host + disc floor)
    curves = 1.0 + 0.3 * (curves - 1.0)
    sampled = MockCampaign(cadence_days=1.0, duration_days=a.days, snr=a.snr, seed=a.seed + 1).sample(
        t, curves, mean_flux)
    paths = write_pycream2(out, list(BANDS), wav, sampled, units="mJy")
    rs.save(out / "true_responses.npz")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, (name, (tt, y, e)) in enumerate(zip(BANDS, sampled)):
        ax[0].errorbar(tt, y / np.mean(y) + 0.15 * k, e / np.mean(y), fmt=".", ms=2, label=name)
        ax[1].plot(rs.tau, rs.psi[k], label=f"{name}: <tau> {rs.mean_lag()[k]:.2f} d")
    ax[0].set(xlabel="t (days)", ylabel="normalised flux + offset", title=f"mock campaign ({a.geometry} disc)")
    ax[1].set(xlabel="delay (days)", ylabel="psi", xlim=(0, 8), title="true response functions")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "mock_campaign.png", dpi=120)

    cmd = pycream2_command(list(BANDS), wav, [f"$MOCK/{p.name}" for p in paths], M_BH, title=f"discsim_{a.geometry}")
    (out / "fit_with_pycream2.sh").write_text(
        "#!/bin/sh\n# usage: cd <echofit repo> && sh <this file>\n"
        'MOCK="$(cd "$(dirname "$0")" && pwd)"\n' + cmd + "\n")
    print("true mean lags (d):", dict(zip(BANDS, np.round(rs.mean_lag(), 2))))
    print("wrote", out, "\nfit with:\n ", cmd)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", choices=["rim", "flat"], default="rim")
    p.add_argument("--inclination", type=float, default=30.0)
    p.add_argument("--days", type=float, default=180.0)
    p.add_argument("--snr", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/mock_rim")
    main(p.parse_args())

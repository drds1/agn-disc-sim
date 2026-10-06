"""
An SPH disc with an embedded massive body (a star, a stellar-mass black-hole
cluster, an intermediate-mass black hole...), against the same disc without
it: does the wake change the reprocessing surface enough to move the delays?

The body launches spiral density waves at once and starts opening a gap over
tens of orbits if it exceeds the thermal mass ``q > (H/R)^3``. The default run
is short (one and a half orbits of the body), enough for the wakes; pass
``--orbits`` for longer runs (and more particles). The reprocessing surface is
the photosphere for electron-scattering opacity, so wakes raise it and a gap
punches a hole in it.

Resolution warning: at the default 8000 particles the SPH smoothing length is
larger than the disc's scale height, so vertical structure is under-resolved.
It demonstrates the pipeline; science runs want 10^6+ particles (Phantom).

    python scripts/embedded_body.py [--n 8000] [--q 5e-3] [--orbits 1.5] [--out outputs/embedded_body] [--reuse]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discsim import (CentralPotential, DiscPhysics, MassiveBodies, Observer, Simulation, SPHGas, SPHSettings,
                     Units, thin_disc)
from discsim import DiscSurface, Snapshot, irradiate, response_functions
from discsim.pipeline import snapshot_surface

M_BH = 5.0e7
R_JOIN = 1.3   # light-days: analytic disc inside, simulated disc outside
H_R, FLARING = 0.05, 0.25   # H/R at 1 ld and its flaring index: a lamppost only sees a flaring surface
WAV = np.array([1500.0, 3000.0, 5100.0, 8000.0])


def run(q: float, n: int, orbits: float, r_body: float, seed: int = 3):
    pos, vel, m = thin_disc(n, r_in=1.0, r_out=4.0, aspect_ratio=H_R, flaring_index=FLARING, sigma_slope=1.0,
                            disc_mass=1e-4, gas=True, seed=seed)
    bodies = MassiveBodies.central()
    if q > 0:
        bodies.add_circular(r_body, q, softening=0.5 * r_body * (q / 3) ** (1 / 3), name="embedded")
    sim = Simulation(bodies, pos, vel, m, potential=CentralPotential(), gas=SPHGas(SPHSettings(aspect_ratio=H_R, flaring_index=FLARING)),
                     r_accrete=0.8, eta_orbit=0.05)
    t_end = orbits * 2 * np.pi * r_body**1.5
    clock = time.time()
    snap = sim.run(t_end, verbose=False)[-1]
    print(f"  q = {q:g}: {len(snap.pos)} particles left, {time.time() - clock:.0f} s")
    return snap


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    units = Units(M_BH)
    phys = DiscPhysics(m_bh=M_BH, mdot=0.05, l_lamppost=3e43, h_lamppost_rg=10.0)
    obs = Observer(30.0, 75.0)
    surface_kw = dict(n_r=20, n_phi=32, kappa=0.4, min_count=6, smooth_phi=5, smooth_r=3)

    results = {}
    for q in (0.0, a.q):
        path = out / f"snap_q{q:g}.npz"
        if a.reuse and path.exists():
            snap = Snapshot.load(path)
        else:
            snap = run(q, a.n, a.orbits, a.r_body)
            snap.save(path)
        outer = snapshot_surface(snap, units, R_JOIN, 4.0, **surface_kw)
        # the simulation has no gas inside ~1 ld; put an analytic disc there with the
        # simulated disc's innermost H/R, so the UV comes from where it should
        h_in = np.nanmedian((outer.z / outer.r[:, None])[:2][outer.valid[:2]])
        inner = DiscSurface.from_function(lambda R, P: h_in * R * (R / R_JOIN) ** FLARING, phys.r_in_ld, R_JOIN,
                                          n_r=60, n_phi=surface_kw["n_phi"])
        surf = DiscSurface.stack(inner, outer)
        disc = irradiate(surf, phys)
        rs = response_functions(disc, obs, WAV, tau_max=10.0, n_tau=300, smooth_days=0.05)
        results[q] = (snap, surf, disc, rs)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for row, (q, (snap, surf, disc, rs)) in enumerate(results.items()):
        p = snap.pos - snap.body_pos[0]
        ax[row, 0].scatter(p[:, 0], p[:, 1], s=0.3, c="k", alpha=0.4)
        if len(snap.body_pos) > 1:
            b = snap.body_pos[1] - snap.body_pos[0]
            ax[row, 0].plot(b[0], b[1], "r*", ms=12)
        ax[row, 0].set(aspect="equal", title=f"particles, q = {q:g}", xlabel="x (ld)", ylabel="y (ld)")
        R, Ph = surf.mesh
        Re, Pe = np.meshgrid(surf.r_edges, surf.phi_edges, indexing="ij")
        zr = np.where(surf.valid, surf.z / R, np.nan)
        zr = np.where(disc.lit, zr, np.nan)
        pc = ax[row, 1].pcolormesh(Re * np.cos(Pe), Re * np.sin(Pe), zr, shading="flat", cmap="viridis")
        fig.colorbar(pc, ax=ax[row, 1], label="photosphere height / R")
        ax[row, 1].set(aspect="equal", title="lit photosphere (white: shadowed or thin)", xlim=(-4.2, 4.2),
                       ylim=(-4.2, 4.2))
        for k, w in enumerate(WAV):
            ax[row, 2].plot(rs.tau, rs.psi[k], label=f"{w:.0f} A, <tau> = {rs.mean_lag()[k]:.2f} d")
        ax[row, 2].set(xlabel="delay (days)", ylabel="psi", title="response functions", xlim=(0, 6))
        ax[row, 2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "embedded_body.png", dpi=120)
    lag0, lag1 = results[0.0][3].mean_lag(), results[a.q][3].mean_lag()
    for w, l0, l1 in zip(WAV, lag0, lag1):
        print(f"  {w:6.0f} A  <tau> no body {l0:5.2f} d   with body {l1:5.2f} d   ratio {l1 / l0:4.2f}")
    print("wrote", out / "embedded_body.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--q", type=float, default=5e-3)
    p.add_argument("--orbits", type=float, default=1.5)
    p.add_argument("--r-body", type=float, default=2.2)
    p.add_argument("--out", default="outputs/embedded_body")
    p.add_argument("--reuse", action="store_true", help="reuse saved snapshots in --out instead of re-running")
    main(p.parse_args())

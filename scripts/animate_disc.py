"""
Animate an SPH disc with an embedded massive body, together with what the
lamppost and the observer make of it at each moment: the lit photosphere and
the continuum response functions.

Panels per frame:
- face-on surface density, with the body's position;
- side view (x-z, vertical axis stretched) showing the disc's thickness and wakes;
- the electron-scattering photosphere, coloured by height/R where the lamppost
  lights it and dark where it is shadowed or optically thin;
- response functions at three wavelengths, against the same disc at t = 0.

Each frame is a frozen snapshot: light-travel delays (days) are much shorter
than the orbital times (months to years), so the response is recomputed per
snapshot, as everywhere else in discsim.

Resolution warning: as in ``embedded_body.py``, a few thousand particles
under-resolve the vertical structure, so frame-to-frame wobbles in the
responses are mostly particle noise. The animation shows the pipeline and the
dynamics, not a measured lag change.

    python scripts/animate_disc.py [--n 8000] [--q 5e-3] [--orbits 4] [--frames 120]
                                   [--out outputs/animation] [--gif docs/images/disc_animation.gif] [--reuse]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from discsim import (CentralPotential, DiscPhysics, DiscSurface, MassiveBodies, Observer, Simulation, Snapshot,
                     SPHGas, SPHSettings, Units, irradiate, response_functions, thin_disc)
from discsim.pipeline import snapshot_surface

M_BH = 5.0e7
R_JOIN = 1.3                 # light-days: analytic disc inside, simulated disc outside
H_R, FLARING = 0.05, 0.25
WAV = np.array([1500.0, 5100.0, 8000.0])
COLOURS = ["#6fa8ff", "#7ee08a", "#ff7a6b"]
EXTENT = 4.6
Z_SIDE = 0.32


def simulate(a, snapdir: Path) -> list[Path]:
    pos, vel, m = thin_disc(a.n, r_in=1.0, r_out=4.0, aspect_ratio=H_R, flaring_index=FLARING, sigma_slope=1.0,
                            disc_mass=1e-4, gas=True, seed=3)
    bodies = MassiveBodies.central()
    bodies.add_circular(a.r_body, a.q, softening=0.5 * a.r_body * (a.q / 3) ** (1 / 3), name="embedded")
    sim = Simulation(bodies, pos, vel, m, potential=CentralPotential(),
                     gas=SPHGas(SPHSettings(aspect_ratio=H_R, flaring_index=FLARING)), r_accrete=0.8,
                     eta_orbit=0.05)
    t_end = a.orbits * 2 * np.pi * a.r_body**1.5
    clock = time.time()
    sim.run(t_end, snapshot_every=t_end / a.frames, outdir=snapdir, verbose=True)
    print(f"simulation: {time.time() - clock:.0f} s")
    return sorted(snapdir.glob("snap_*.npz"))


def reprocess(snap: Snapshot, units: Units, phys: DiscPhysics, obs: Observer):
    outer = snapshot_surface(snap, units, R_JOIN, 4.0, n_r=20, n_phi=32, kappa=0.4, min_count=6, smooth_phi=5,
                             smooth_r=3)
    # no gas inside ~1 ld: an analytic disc there with the simulated disc's innermost H/R
    h_in = np.nanmedian((outer.z / outer.r[:, None])[:2][outer.valid[:2]])
    inner = DiscSurface.from_function(lambda R, P: h_in * R * (R / R_JOIN) ** FLARING, phys.r_in_ld, R_JOIN,
                                      n_r=60, n_phi=32)
    disc = irradiate(DiscSurface.stack(inner, outer), phys)
    rs = response_functions(disc, obs, WAV, tau_max=8.0, n_tau=240, smooth_days=0.08)
    return disc, rs


def column_density(x, y, m, extent, bins, sigma=1.2):
    img, _, _ = np.histogram2d(x, y, bins=bins, range=extent, weights=m)
    return gaussian_filter(img, sigma).T


def render(paths: list[Path], frame_dir: Path, a):
    units = Units(M_BH)
    phys = DiscPhysics(m_bh=M_BH, mdot=0.05, l_lamppost=3e43, h_lamppost_rg=10.0)
    obs = Observer(30.0, 75.0)
    snaps = [Snapshot.load(p) for p in paths]
    reproc = [reprocess(s, units, phys, obs) for s in snaps]
    lags = np.array([rs.mean_lag() for _, rs in reproc])
    rs0 = reproc[0][1]
    period_yr = units.orbital_period_days(a.r_body) / 365.25

    # fixed colour scales across frames
    sig0 = column_density(*(snaps[0].pos - snaps[0].body_pos[0])[:, :2].T, snaps[0].mass,
                          [[-EXTENT, EXTENT]] * 2, 260, sigma=1.5)
    vmax_s = np.percentile(sig0[sig0 > 0], 99.7)
    norm_s = LogNorm(vmax_s / 300, vmax_s)

    plt.style.use("dark_background")
    for k, (snap, (disc, rs)) in enumerate(zip(snaps, reproc)):
        fig = plt.figure(figsize=(13, 7.4), dpi=a.dpi, facecolor="#0b0d12")
        ax_face = fig.add_axes([0.045, 0.35, 0.29, 0.55])
        ax_phot = fig.add_axes([0.355, 0.35, 0.29, 0.55])
        ax_resp = fig.add_axes([0.765, 0.40, 0.225, 0.48])
        ax_side = fig.add_axes([0.06, 0.08, 0.93, 0.19])
        for ax in (ax_face, ax_phot, ax_resp, ax_side):
            ax.set_facecolor("#0b0d12")

        p = snap.pos - snap.body_pos[0]
        b = snap.body_pos[1] - snap.body_pos[0]
        sig = column_density(p[:, 0], p[:, 1], snap.mass, [[-EXTENT, EXTENT]] * 2, 260, sigma=1.5)
        ax_face.imshow(np.where(sig > 0, sig, np.nan), origin="lower", extent=[-EXTENT, EXTENT] * 2,
                       cmap="magma", norm=norm_s, interpolation="bilinear")
        ax_face.plot(0, 0, "o", color="white", ms=4)
        ax_face.plot(b[0], b[1], "*", color="#7fe7ff", ms=13, mec="black", mew=0.6)
        ax_face.set(aspect="equal", xlabel="x (light-days)", ylabel="y (light-days)",
                    title="gas surface density (face-on)")

        side = column_density(p[:, 0], p[:, 2], snap.mass, [[-EXTENT, EXTENT], [-Z_SIDE, Z_SIDE]], [260, 70],
                              sigma=1.0)
        ax_side.imshow(np.where(side > 0, side, np.nan), origin="lower", extent=[-EXTENT, EXTENT, -Z_SIDE, Z_SIDE],
                       cmap="magma", norm=LogNorm(side.max() / 300, side.max()), aspect="auto",
                       interpolation="bilinear")
        ax_side.plot(b[0], b[2], "*", color="#7fe7ff", ms=11, mec="black", mew=0.6)
        ax_side.plot(0, phys.h_lp_ld, "o", color="white", ms=3)
        box = ax_side.get_position()
        stretch = (box.height * fig.get_figheight() / (2 * Z_SIDE)) / (box.width * fig.get_figwidth() / (2 * EXTENT))
        ax_side.set(xlabel="x (light-days)", ylabel="z (ld)",
                    title=f"side view (vertical scale stretched ×{stretch:.1f})")

        surf = disc.surface
        Re, Pe = np.meshgrid(surf.r_edges, surf.phi_edges, indexing="ij")
        zr = np.where(surf.valid & disc.lit, surf.z / surf.r[:, None], np.nan)
        pc = ax_phot.pcolormesh(Re * np.cos(Pe), Re * np.sin(Pe), zr, shading="flat", cmap="viridis",
                                vmin=0.05, vmax=0.32)
        ax_phot.plot(b[0], b[1], "*", color="#7fe7ff", ms=9, mec="black", mew=0.6)
        cb = fig.colorbar(pc, cax=fig.add_axes([0.655, 0.40, 0.009, 0.45]))
        cb.set_label("height / R", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax_phot.set(aspect="equal", xlim=(-4.2, 4.2), ylim=(-4.2, 4.2),
                    title="lit photosphere (dark: shadowed or thin)", xlabel="x (light-days)")
        ax_phot.tick_params(labelsize=7)

        for j, (w, c) in enumerate(zip(WAV, COLOURS)):
            ax_resp.plot(rs0.tau, rs0.psi[j], color=c, lw=0.8, ls=":", alpha=0.7)
            ax_resp.plot(rs.tau, rs.psi[j], color=c, lw=1.6,
                         label=f"{w:.0f} Å: mean lag {lags[k, j]:.2f} d")
        ax_resp.set(xlim=(0, 5), ylim=(0, 4.6), xlabel="delay τ (days)", ylabel="ψ(τ)",
                    title="response functions (dotted: t = 0)")
        ax_resp.legend(fontsize=8, frameon=False, loc="upper right")
        ax_resp.tick_params(labelsize=7)

        t_yr = snap.t * units.time_days / 365.25
        fig.suptitle(f"SPH disc + embedded body (q = {a.q:g}) around a {M_BH:.0e} M$_\\odot$ black hole"
                     f"      t = {t_yr:4.2f} yr  ({t_yr / period_yr:4.2f} orbits of the body)",
                     fontsize=12, color="#e8e8e8", y=0.975)
        fig.savefig(frame_dir / f"frame_{k:04d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"rendered {len(snaps)} frames")
    for w, l0, l1 in zip(WAV, lags[0], lags[-1]):
        print(f"  {w:6.0f} A  <tau> start {l0:5.2f} d  end {l1:5.2f} d  "
              f"(frame-to-frame scatter {np.std(np.diff(lags[:, list(WAV).index(w)])):.2f} d)")


def make_gif(frame_dir: Path, gif: Path, fps: int, width: int):
    gif.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        # particle noise makes every frame different, so size is set by frame count, width and palette
        filt = f"fps={fps},scale={width}:-1:flags=lanczos"
        src = ["-framerate", "15", "-i", str(frame_dir / "frame_%04d.png")]
        palette = frame_dir / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-vf",
                        f"{filt},palettegen=max_colors=96:stats_mode=diff", str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-i", str(palette), "-lavfi",
                        f"{filt} [x]; [x][1:v] paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", str(gif)],
                       check=True)
    else:
        from PIL import Image
        frames = [Image.open(p) for p in sorted(frame_dir.glob("frame_*.png"))]
        frames = [f.resize((width, int(f.height * width / f.width))) for f in frames]
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


def main(a):
    out = Path(a.out)
    snapdir, frame_dir = out / "snaps", out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(snapdir.glob("snap_*.npz")) if a.reuse else []
    if not paths:
        paths = simulate(a, snapdir)
    for f in frame_dir.glob("frame_*.png"):
        f.unlink()
    render(paths, frame_dir, a)
    make_gif(frame_dir, Path(a.gif), a.fps, a.width)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--q", type=float, default=5e-3)
    p.add_argument("--r-body", type=float, default=2.2)
    p.add_argument("--orbits", type=float, default=4.0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=8, help="GIF frame rate (frames are dropped from 15 fps)")
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--width", type=int, default=720, help="GIF width in pixels")
    p.add_argument("--out", default="outputs/animation")
    p.add_argument("--gif", default="docs/images/disc_animation.gif")
    p.add_argument("--reuse", action="store_true", help="reuse saved snapshots in --out instead of re-running")
    main(p.parse_args())

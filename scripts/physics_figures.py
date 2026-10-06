"""
Figures for ``docs/physics.md``: timescales, the SPH kernel and density
estimate, integrator energy errors, the lamppost geometry with shadows and
isodelay surfaces, and where each wavelength's response comes from.

    python scripts/physics_figures.py [--out docs/images]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfcinv

from discsim import DiscPhysics, DiscSurface, Observer, Units, irradiate, thin_disc_lag
from discsim.reverb import dplanck_dT, visibility

# validated categorical order (blue, orange, aqua, yellow, magenta); lines are direct-labelled
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

M_BH = 5.0e7


def label_end(ax, x, y, text, colour, dx=1.04, va="center"):
    ax.annotate(text, (x[-1], y[-1]), xytext=(x[-1] * dx, y[-1]), color=INK, fontsize=9, va=va,
                annotation_clip=False)
    ax.plot(x[-1], y[-1], "o", color=colour, ms=4)


def timescales(out: Path):
    u = Units(M_BH)
    R = np.logspace(np.log10(10 * u.r_g_ld), np.log10(30), 200)    # light-days
    alpha, h_r = 0.1, 0.01
    t_orb = u.orbital_period_days(R)
    t_light = R
    t_th = t_orb / (2 * np.pi * alpha)
    t_visc = t_th / h_r**2
    lag = thin_disc_lag([1500, 8000], M_BH, 0.05)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axvspan(lag[0], lag[1], color=C[2], alpha=0.12, lw=0)
    ax.text(np.sqrt(lag[0] * lag[1]), 3e-3, "UV–optical\nreprocessing\nradii", ha="center", fontsize=8.5,
            color=INK)
    rows = [(t_light, "light travel  R/c", C[0]), (t_orb, "orbit  2π/Ω", C[1]),
            (t_th, f"thermal  1/(αΩ), α = {alpha}", C[2]), (t_visc, f"viscous  (R/H)²/(αΩ), H/R = {h_r}", C[4])]
    for (y, lab, c), nudge in zip(rows, [1, 0.5, 2.2, 1]):
        ax.plot(R, y, color=c)
        ax.plot(R[-1], y[-1], "o", color=c, ms=4)
        ax.text(R[-1] * 1.08, y[-1] * nudge, lab, fontsize=9, va="center", color=INK)
    for yr, lab in [(1, "1 day"), (365.25, "1 yr"), (365.25e3, "1000 yr")]:
        ax.axhline(yr, color=MUTED, lw=0.6, ls=":")
        ax.text(25, yr * 1.4, lab, fontsize=8, color=MUTED, ha="right")
    ax.set(xscale="log", yscale="log", xlabel="radius R (light-days)", ylabel="timescale (days)",
           ylim=(1e-3, 1e9), xlim=(R[0], R[-1]))
    fig.suptitle(f"Timescales in a disc around 5×10$^7$ M$_\\odot$", x=0.02, ha="left", fontweight="bold", fontsize=11)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / u.r_g_ld, lambda x: x * u.r_g_ld))
    sec.set_xlabel("R (r$_g$)", color=MUTED)
    fig.subplots_adjust(right=0.64, top=0.82)
    fig.savefig(out / "physics_timescales.png", dpi=150)
    plt.close(fig)


def w1d(q):
    return (2 / 3) * np.where(q < 1, 1 - 1.5 * q**2 + 0.75 * q**3, np.where(q < 2, 0.25 * (2 - q) ** 3, 0))


def sph_kernel(out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    q = np.linspace(0, 2.5, 300)
    w = np.where(q < 1, 1 - 1.5 * q**2 + 0.75 * q**3, np.where(q < 2, 0.25 * (2 - q) ** 3, 0)) / np.pi
    dw = np.where(q < 1, -3 * q + 2.25 * q**2, np.where(q < 2, -0.75 * (2 - q) ** 2, 0)) / np.pi
    ax[0].plot(q, w, color=C[0])
    ax[0].plot(q, dw, color=C[1])
    ax[0].text(0.35, 0.29, "W(q) h³", color=INK, fontsize=9)
    ax[0].text(1.1, -0.22, "dW/dq h⁴", color=INK, fontsize=9)
    ax[0].axvline(2, color=MUTED, lw=0.8, ls=":")
    ax[0].text(2.03, 0.2, "support\nq = 2", fontsize=8, color=MUTED)
    ax[0].set(xlabel="q = r / h", title="Cubic-spline kernel (3D)")

    rng = np.random.default_rng(1)
    n, m = 200, 1 / 200
    x = rng.normal(0, 1, n)
    xs = np.linspace(-4, 4, 400)
    ax[1].plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), color=MUTED, lw=1.2, ls="--")
    for k, (n_ngb, c) in enumerate([(5, C[1]), (50, C[0])]):
        d = np.sort(np.abs(x[None, :] - x[:, None]), axis=1)
        h = 0.5 * d[:, n_ngb]
        rho = (m * w1d(np.abs(xs[:, None] - x[None, :]) / h[None, :]) / h[None, :]).sum(axis=1)
        ax[1].plot(xs, rho, color=c, lw=1.6)
        ax[1].text(2.0, 0.42 - 0.06 * k, f"{n_ngb} neighbours", fontsize=9, color=INK)
        ax[1].plot([1.6, 1.9], [0.425 - 0.06 * k] * 2, color=c, lw=2)
    ax[1].plot(x, np.full(n, -0.02), "|", color=INK, ms=8, alpha=0.5)
    ax[1].text(-3.9, 0.55, "dashed: true density\nticks: particles", fontsize=8, color=MUTED, va="top")
    ax[1].set(xlabel="x", ylabel="ρ", title="Density as a sum of kernels (1D)", ylim=(-0.05, 0.6))
    fig.tight_layout()
    fig.savefig(out / "physics_sph_kernel.png", dpi=150)
    plt.close(fig)


def integrators(out: Path):
    e, n_orb, n_per = 0.6, 60, 200
    dt = 2 * np.pi / n_per

    def acc(x):
        return -x / np.linalg.norm(x) ** 3

    def energy(x, v):
        return 0.5 * v @ v - 1 / np.linalg.norm(x)

    x0 = np.array([1 - e, 0.0])
    v0 = np.array([0.0, np.sqrt((1 + e) / (1 - e))])
    e0 = energy(x0, v0)
    steps = n_orb * n_per
    res = {}
    for name in ("Euler", "RK4", "leapfrog (KDK)"):
        x, v = x0.copy(), v0.copy()
        err, path = np.empty(steps), np.empty((steps, 2))
        for s in range(steps):
            if name == "Euler":
                x, v = x + dt * v, v + dt * acc(x)
            elif name == "RK4":
                k1x, k1v = v, acc(x)
                k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x)
                k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x)
                k4x, k4v = v + dt * k3v, acc(x + dt * k3x)
                x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
                v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            else:
                v = v + 0.5 * dt * acc(x)
                x = x + dt * v
                v = v + 0.5 * dt * acc(x)
            err[s] = abs(energy(x, v) / e0 - 1)
            path[s] = x
        res[name] = (err, path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[1, 1.5]))
    t = np.arange(1, steps + 1) / n_per
    for (name, (err, path)), c in zip(res.items(), [C[1], C[0], C[2]]):
        ax[0].plot(path[:, 0], path[:, 1], color=c, lw=0.6 if name != "Euler" else 0.4)
        ax[1].plot(t, np.maximum(err, 1e-16), color=c, lw=1.2)
        ax[1].text(t[-1] * 1.02, max(err[-1], 1e-16), name, fontsize=9, color=INK, va="center")
    ax[0].plot(0, 0, "o", color=INK, ms=5)
    ax[0].set(aspect="equal", xlim=(-3, 1.5), ylim=(-2, 2), title=f"Orbits, e = {e}, {n_per} steps/orbit")
    ax[0].text(-2.9, -1.9, "Euler spirals outwards", fontsize=8, color=MUTED)
    ax[1].set(yscale="log", xlabel="time (orbits)", ylabel="|ΔE / E|", xlim=(0, n_orb), ylim=(1e-9, 10),
              title="Energy error")
    fig.subplots_adjust(left=0.06, right=0.86, wspace=0.25, bottom=0.13)
    fig.savefig(out / "physics_integrators.png", dpi=150)
    plt.close(fig)


def lamppost_geometry(out: Path):
    phys = DiscPhysics(m_bh=M_BH, mdot=0.05, l_lamppost=3e43, h_lamppost_rg=40.0)
    # illustrative: a mildly flared disc with a puffed-up ring at 2.5 ld that shadows the disc behind it
    surf = DiscSurface.from_function(lambda R, P: 0.03 * R * R**0.25 + 0.25 * np.exp(-0.5 * ((R - 2.5) / 0.3) ** 2),
                                     phys.r_in_ld, 7.0, n_r=400, n_phi=2)
    disc = irradiate(surf, phys)
    R, z, lit = surf.r, surf.z[:, 0], disc.lit[:, 0]
    h = phys.h_lp_ld
    inc = np.radians(30.0)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    xx, zz = np.meshgrid(np.linspace(-7.5, 7.5, 600), np.linspace(-0.5, 1.1, 300))
    tau = np.hypot(xx, zz - h) - (xx * np.sin(inc) + (zz - h) * np.cos(inc))
    cs = ax.contour(xx, zz, tau, levels=[1, 2, 4, 6, 8, 10], colors=MUTED, linewidths=0.7, linestyles="--")
    ax.clabel(cs, fmt=lambda v: f"τ = {v:g} d", fontsize=7.5)
    for sgn in (1, -1):
        x = sgn * R
        ax.fill_between(x, -z, z, color="#d9d7cf", lw=0)
        seg = np.where(lit, z, np.nan)
        ax.plot(x, seg, color=C[3], lw=2.5)
        ax.plot(x, np.where(~lit, z, np.nan), color=C[0], lw=2.5)
    k = np.argmax(np.where(R < 4, (z - h) / R, -np.inf))
    for sgn in (1, -1):
        ax.plot([0, sgn * 7.2], [h, h + (z[k] - h) * 7.2 / R[k]], color=C[3], lw=0.8, ls="-", alpha=0.8)
    ax.plot(0, h, "*", color=C[3], ms=14, mec=INK, mew=0.5)
    ax.annotate("lamppost", (0, h), xytext=(0.4, 0.55), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate("lit surface", (1.5, z[np.searchsorted(R, 1.5)]), xytext=(1.0, 0.75), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate("in the rim's shadow", (4.5, z[np.searchsorted(R, 4.5)]), xytext=(4.2, 0.85), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    # observer direction (sin i, 0, cos i): towards +x and up (steeper on screen because of the stretch)
    tail = np.array([-5.6, 0.5])
    ax.annotate("", xy=tail + 0.5 * np.array([np.sin(inc), np.cos(inc)]), xytext=tail,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(tail[0] - 0.15, tail[1] + 0.2, "to observer\n(i = 30°)", fontsize=9, ha="right", color=INK)
    ax.text(7.3, -0.45, "isodelay surfaces (dashed): paraboloids of constant lamp → disc → observer delay",
            fontsize=8, color=MUTED, ha="right")
    ax.set(xlim=(-7.5, 7.5), ylim=(-0.5, 1.1), xlabel="x (light-days)", ylabel="z (light-days)",
           title="A disc with a puffed-up ring, side-on (vertical scale stretched ×4)")
    fig.tight_layout()
    fig.savefig(out / "physics_lamppost.png", dpi=150)
    plt.close(fig)


def response_origin(out: Path):
    phys = DiscPhysics(m_bh=M_BH, mdot=0.05, l_lamppost=3e43, h_lamppost_rg=10.0)
    surf = DiscSurface.flat(phys.r_in_ld, 40.0, n_r=600, n_phi=8)
    disc = irradiate(surf, phys)
    obs = Observer(0.0)
    proj = visibility(disc, obs)
    t = disc.temp
    dtdl = np.where(disc.lit & (t > 0), disc.t_irr**4 / (4 * np.maximum(t, 1e-3) ** 3), 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for j, (wav, c) in enumerate(zip([1500, 3000, 5100, 8000], C)):
        w = (dplanck_dT(wav, t) * dtdl * proj).sum(axis=1)
        dlnr = np.gradient(np.log(surf.r))
        y = w / dlnr
        y /= y.max()
        ax[0].plot(surf.r, y, color=c)
        k = np.argmax(y)
        ax[0].text(surf.r[k], 1.03 + 0.07 * (j % 2), f"{wav} Å", ha="center", fontsize=9, color=INK)
    ax[0].set(xscale="log", xlim=(0.03, 40), ylim=(0, 1.2), xlabel="R (light-days)",
              ylabel="response per ln R (peak = 1)", title="Where each band's variable light comes from")

    ks = np.logspace(0.31, 6.5, 300)
    ax[1].plot(ks, np.sqrt(2) * erfcinv(2 / ks), color=C[0])
    for v, lab in [(1e4, "10⁴"), (1e6, "10⁶")]:
        ax[1].plot(v, np.sqrt(2) * erfcinv(2 / v), "o", color=C[0], ms=6, mec=SURFACE, mew=1.5)
        ax[1].text(v, np.sqrt(2) * erfcinv(2 / v) - 0.45, f"κΣ = {lab}", fontsize=8.5, ha="center")
    ax[1].axvline(2, color=MUTED, lw=0.8, ls=":")
    ax[1].text(2.3, 0.25, "optically thin:\nno surface", fontsize=8, color=MUTED)
    ax[1].set(xscale="log", xlabel="κΣ (vertical optical depth through the disc)",
              ylabel="photosphere height / σ$_z$", ylim=(0, 5.5),
              title="Photosphere of a Gaussian layer: τ(z) = 1")
    fig.tight_layout()
    fig.savefig(out / "physics_response_origin.png", dpi=150)
    plt.close(fig)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (timescales, sph_kernel, integrators, lamppost_geometry, response_origin):
        f(out)
        print("done", f.__name__)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/images")
    main(p.parse_args())

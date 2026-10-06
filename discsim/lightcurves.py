"""
Mock light curves from simulated response functions, written in the format
pycream2's ``scripts/fit_lightcurves.py`` reads (``t y yerr`` with a comment
header), so a simulated disc can be fitted with the standard flat-disc model and
the bias measured: does a rimmed, warped or perturbed disc fool a thin-disc fit?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reverb import ResponseSet


def drw_driver(t_span: float, dt: float = 0.05, tau_drw: float = 100.0, sigma: float = 0.2,
               seed: int = 0):
    """Damped random walk (exact Ornstein-Uhlenbeck updates) on a regular grid.
    Returns ``t, x`` with ``x`` a fractional variation about 1."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(t_span / dt)) + 1
    t = np.arange(n) * dt
    a = np.exp(-dt / tau_drw)
    s = sigma * np.sqrt(1.0 - a**2)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    eps = rng.normal(0, 1, n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + s * eps[k]
    return t, 1.0 + x


def echo(t_driver: np.ndarray, driver: np.ndarray, responses: ResponseSet) -> np.ndarray:
    """Convolve the driver with each band's response; returns (n_band, n_t).
    Times before the response's full support are padded with the first value."""
    dt = t_driver[1] - t_driver[0]
    out = []
    for psi in responses.psi:
        kern_t = np.arange(0.0, responses.tau[-1] + dt, dt)
        kern = np.interp(kern_t, responses.tau, psi, left=psi[0], right=0.0) * dt
        kern /= kern.sum()
        pad = np.concatenate([np.full(len(kern) - 1, driver[0]), driver])
        out.append(np.convolve(pad, kern, mode="valid"))
    return np.array(out)


@dataclass
class MockCampaign:
    cadence_days: float = 1.0
    duration_days: float = 200.0
    snr: float = 100.0
    gap_fraction: float = 0.1
    seed: int = 1

    def sample(self, t_fine, curves, mean_flux=None):
        rng = np.random.default_rng(self.seed)
        n = int(self.duration_days / self.cadence_days)
        t0 = t_fine[-1] - self.duration_days
        out = []
        for b, c in enumerate(curves):
            t = t0 + np.sort(rng.uniform(0, self.duration_days, n))
            t = t[rng.uniform(size=n) > self.gap_fraction]
            y = np.interp(t, t_fine, c) * (1.0 if mean_flux is None else mean_flux[b])
            err = np.abs(y) / self.snr
            out.append((t - t0, y + rng.normal(0, err), err))
        return out


def write_pycream2(dirpath, names, wavelengths, sampled, units: str = "arbitrary") -> list[Path]:
    """Write one ``<name>.txt`` per band in pycream2's text format."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, wav, (t, y, e) in zip(names, wavelengths, sampled):
        p = d / f"{name}.txt"
        header = f"t(days) y yerr -- band '{name}', wavelength {wav:.1f} A (simulated, flux {units})"
        np.savetxt(p, np.column_stack([t, y, e]), fmt="%.6f", header=header)
        paths.append(p)
    return paths


def pycream2_command(names, wavelengths, paths, m_bh: float, title: str = "discsim_mock") -> str:
    bands = " ".join(f"--band {n} {w:.0f} {p}" for n, w, p in zip(names, wavelengths, paths))
    return f"python scripts/fit_lightcurves.py --title {title} --m-bh {m_bh:.3g} {bands}"

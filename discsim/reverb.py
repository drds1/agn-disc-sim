"""
Lamppost irradiation, temperature, and reverberation response of a disc
surface: the step that turns a simulated disc into delays, response functions
and an SED you can put next to pycream2's fits.

Geometry (light-days, days): the lamppost sits on the spin axis at height
``h_lp`` above the central black hole; the observer lies in the x-z plane at
inclination ``i`` from +z, direction ``o = (sin i, 0, cos i)``, so azimuth 0
faces the observer.

Per surface element:

- **Shadowing.** With the lamp on the axis, light travels in planes of constant
  azimuth, so an element is lit if its elevation seen from the lamp is at least
  that of every element inside it at the same azimuth (a horizon scan), and its
  normal faces the lamp.
- **Temperature.** ``T^4 = T_visc^4 + T_irr^4`` with
  ``sigma T_visc^4 = 3 G M Mdot / (8 pi R^3) (1 - sqrt(R_in / R))`` and
  ``sigma T_irr^4 = (1 - A) L_lp cos(theta_inc) / (4 pi d^2)``.
- **Delay.** ``c tau = d - o . (x - x_lp)``: lamp to element to observer, minus
  lamp straight to observer.
- **Response.** A small change ``dL`` in the lamp changes the element's
  ``B_nu(T)`` by ``dB_nu/dT * T_irr^4 / (4 T^3 L) * dL``, seen through its
  projected area ``(n . o) dA``. Histogramming those weights in ``tau`` gives
  ``psi(tau | lambda)``, normalised to unit area.

Not yet included (roadmap): occultation of one part of the disc by another
along the line of sight; the underside of a warped disc; relativistic light
bending and Doppler shifts near the black hole; a colour correction or real
atmosphere instead of a blackbody; diffuse continuum from the broad-line
region; a non-linear (temperature-dependent) reprocessing response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface import DiscSurface
from .units import (C_CGS, G_CGS, H_CGS, K_CGS, LIGHT_DAY_CM, MPC_CM, MSUN_G, SIGMA_SB, YEAR_S, Units)


@dataclass
class DiscPhysics:
    m_bh: float = 1.0e8               # Msun
    mdot: float = 0.1                 # Msun / yr
    l_lamppost: float = 1.0e44        # erg / s, the variable irradiating source
    h_lamppost_rg: float = 10.0       # lamppost height, gravitational radii
    albedo: float = 0.0
    r_in_rg: float = 6.0              # inner edge for the viscous term

    @property
    def units(self) -> Units:
        return Units(self.m_bh)

    @property
    def h_lp_ld(self) -> float:
        return self.h_lamppost_rg * self.units.r_g_ld

    @property
    def r_in_ld(self) -> float:
        return self.r_in_rg * self.units.r_g_ld

    @property
    def eddington_ratio(self) -> float:
        return self.mdot / self.units.mdot_edd_msun_yr()


@dataclass
class Observer:
    inclination_deg: float = 30.0
    distance_mpc: float = 100.0

    @property
    def direction(self) -> np.ndarray:
        i = np.radians(self.inclination_deg)
        return np.array([np.sin(i), 0.0, np.cos(i)])


@dataclass
class IrradiatedDisc:
    """Everything the response and SED need, per surface element."""

    surface: DiscSurface
    physics: DiscPhysics
    pos: np.ndarray        # (n_r, n_phi, 3) light-days
    normal: np.ndarray
    area: np.ndarray       # light-days^2
    lit: np.ndarray        # bool
    t_visc: np.ndarray     # K
    t_irr: np.ndarray      # K
    temp: np.ndarray       # K


def shadow_mask(surface: DiscSurface, h_lp: float) -> np.ndarray:
    """True where the lamppost at height ``h_lp`` sees the surface."""
    elev = np.arctan2(surface.z - h_lp, surface.r[:, None])
    elev = np.where(surface.valid, elev, -np.inf)
    horizon = np.maximum.accumulate(elev, axis=0)
    return surface.valid & (elev >= horizon - 1e-12)


def irradiate(surface: DiscSurface, physics: DiscPhysics) -> IrradiatedDisc:
    pos = surface.positions()
    normal = surface.normals()
    area = surface.areas()
    lamp = np.array([0.0, 0.0, physics.h_lp_ld])
    to_lamp = lamp - pos
    d = np.linalg.norm(to_lamp, axis=-1)
    cos_inc = np.sum(normal * to_lamp, axis=-1) / d
    lit = shadow_mask(surface, physics.h_lp_ld) & (cos_inc > 0.0)

    d_cm = d * LIGHT_DAY_CM
    f_irr = np.where(lit, (1.0 - physics.albedo) * physics.l_lamppost * cos_inc / (4.0 * np.pi * d_cm**2), 0.0)
    t_irr = (f_irr / SIGMA_SB) ** 0.25

    R = np.hypot(pos[..., 0], pos[..., 1])
    R_cm = R * LIGHT_DAY_CM
    mdot_cgs = physics.mdot * MSUN_G / YEAR_S
    fac = np.clip(1.0 - np.sqrt(physics.r_in_ld / R), 0.0, None)
    t_visc = (3.0 * G_CGS * physics.m_bh * MSUN_G * mdot_cgs / (8.0 * np.pi * SIGMA_SB * R_cm**3) * fac) ** 0.25
    t_visc = np.where(surface.valid, t_visc, 0.0)
    temp = (t_visc**4 + t_irr**4) ** 0.25
    return IrradiatedDisc(surface, physics, pos, normal, area, lit, t_visc, t_irr, temp)


# ------------------------------------------------------------------ Planck
def planck_nu(wavelength_A, temp):
    """B_nu(T) in cgs per steradian."""
    lam = np.asarray(wavelength_A, float) * 1e-8
    nu = C_CGS / lam
    x = H_CGS * nu / (K_CGS * np.maximum(temp, 1e-3))
    return 2.0 * H_CGS * nu**3 / C_CGS**2 / np.expm1(np.minimum(x, 700.0))


def dplanck_dT(wavelength_A, temp):
    lam = np.asarray(wavelength_A, float) * 1e-8
    nu = C_CGS / lam
    t = np.maximum(temp, 1e-3)
    x = np.minimum(H_CGS * nu / (K_CGS * t), 700.0)
    ex = np.exp(-x)
    return 2.0 * H_CGS * nu**3 / C_CGS**2 * x / t * ex / (1.0 - ex) ** 2


# ---------------------------------------------------------------- response
@dataclass
class ResponseSet:
    tau: np.ndarray            # days, bin centres
    wavelengths: np.ndarray    # Angstrom
    psi: np.ndarray            # (n_wav, n_tau), each row unit area
    observer: Observer

    def mean_lag(self) -> np.ndarray:
        return np.trapezoid(self.psi * self.tau, self.tau, axis=1)

    def median_lag(self) -> np.ndarray:
        dt = np.gradient(self.tau)
        cdf = np.cumsum(self.psi * dt, axis=1)
        return np.array([np.interp(0.5, c / c[-1], self.tau) for c in cdf])

    def lag_slope(self, which: str = "mean") -> float:
        """Best-fit power-law index of lag vs wavelength (4/3 for a standard disc)."""
        lag = self.mean_lag() if which == "mean" else self.median_lag()
        return float(np.polyfit(np.log(self.wavelengths), np.log(lag), 1)[0])

    def save(self, path) -> None:
        np.savez(path, tau=self.tau, wavelengths=self.wavelengths, psi=self.psi,
                 inclination_deg=self.observer.inclination_deg)


def delays(disc: IrradiatedDisc, observer: Observer) -> np.ndarray:
    lamp = np.array([0.0, 0.0, disc.physics.h_lp_ld])
    rel = disc.pos - lamp
    return np.linalg.norm(rel, axis=-1) - rel @ observer.direction


def visibility(disc: IrradiatedDisc, observer: Observer) -> np.ndarray:
    """Projected area towards the observer, light-days^2 (no occultation)."""
    mu = disc.normal @ observer.direction
    return np.where(disc.surface.valid & (mu > 0.0), mu * disc.area, 0.0)


def response_functions(disc: IrradiatedDisc, observer: Observer, wavelengths_A, tau_max: float | None = None,
                       n_tau: int = 400, smooth_days: float = 0.0) -> ResponseSet:
    wav = np.atleast_1d(np.asarray(wavelengths_A, float))
    tau_el = delays(disc, observer).ravel()
    proj = visibility(disc, observer).ravel()
    t = disc.temp.ravel()
    dtdl = np.where(disc.lit.ravel() & (t > 0), disc.t_irr.ravel() ** 4 / (4.0 * np.maximum(t, 1e-3) ** 3), 0.0)
    if tau_max is None:
        tau_max = float(np.max(tau_el[proj > 0])) * 1.02
    edges = np.linspace(0.0, tau_max, n_tau + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    psi = np.zeros((len(wav), n_tau))
    for k, w in enumerate(wav):
        weight = dplanck_dT(w, t) * dtdl * proj
        h, _ = np.histogram(tau_el, bins=edges, weights=weight)
        if smooth_days > 0:
            x = np.arange(-4 * smooth_days, 4 * smooth_days + 1e-12, edges[1] - edges[0])
            g = np.exp(-0.5 * (x / smooth_days) ** 2)
            h = np.convolve(h, g / g.sum(), mode="same")
        area = np.trapezoid(h, centres)
        psi[k] = h / area if area > 0 else h
    return ResponseSet(centres, wav, psi, observer)


def sed(disc: IrradiatedDisc, observer: Observer, wavelengths_A) -> np.ndarray:
    """Observed f_nu (mJy) of the disc's steady emission."""
    wav = np.atleast_1d(np.asarray(wavelengths_A, float))
    proj = visibility(disc, observer) * LIGHT_DAY_CM**2
    d2 = (observer.distance_mpc * MPC_CM) ** 2
    return np.array([np.sum(planck_nu(w, disc.temp) * proj) / d2 for w in wav]) * 1e26


def thin_disc_lag(wavelength_A, m_bh: float, mdot: float, l_lamppost: float = 0.0, albedo: float = 0.0,
                  h_ld: float = 0.0, x: float = 2.49) -> np.ndarray:
    """Analytic face-on thin-disc lag (days), ``tau = R(lambda) / c`` with
    ``k T(R) = h c / (x lambda)`` and ``T^4 = [3 G M Mdot / 8 pi sigma + (1 - A) L h / 4 pi sigma] R^-3``
    (Fausnaugh et al. 2016 eq. 12 form; x = 2.49 is the flux-weighted mean
    radius of a T ~ R^-3/4 disc)."""
    lam = np.asarray(wavelength_A, float) * 1e-8
    k3 = 3 * G_CGS * m_bh * MSUN_G * mdot * MSUN_G / YEAR_S / (8 * np.pi * SIGMA_SB)
    k3 += (1 - albedo) * l_lamppost * h_ld * LIGHT_DAY_CM / (4 * np.pi * SIGMA_SB)
    t_needed = H_CGS * C_CGS / (x * K_CGS * lam)
    return (k3 / t_needed**4) ** (1.0 / 3.0) / LIGHT_DAY_CM

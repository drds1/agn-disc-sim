"""
The disc's reprocessing surface on a polar (R, phi) grid, in light-days.

A surface is a height map ``z(R, phi)`` of the side facing the lamppost (+z),
plus its normals and area elements. It can come from

- an analytic profile (:meth:`DiscSurface.from_function`): flat discs, flared
  discs, the rims and ripples of Starkey et al. (2023), anything ``H(R, phi)``;
- a particle snapshot (:meth:`DiscSurface.from_particles`): an SPH or
  test-particle disc. In each cell the mid-plane is the median particle height
  and the surface sits ``n_sigma`` robust standard deviations above it, a proxy
  for the optical-depth-one photosphere (an isothermal Gaussian disc reaches
  tau = 1 at 2-4 H for AGN surface densities; ``n_sigma`` is that number). A
  proper photosphere needs a density field and an opacity: on the roadmap.

A single-valued ``z(R, phi)`` handles flaring, rims, ripples, spirals, gaps and
moderate warps (tilts up to ~60 degrees). It cannot represent a disc torn into
separate overlapping rings, or the underside of a strongly warped disc that the
lamppost may also see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .units import LIGHT_DAY_CM


@dataclass
class DiscSurface:
    r_edges: np.ndarray          # light-days, length n_r + 1
    phi_edges: np.ndarray        # radians, length n_phi + 1
    z: np.ndarray                # (n_r, n_phi) surface height, light-days
    valid: np.ndarray            # (n_r, n_phi) bool: disc present
    z_mid: np.ndarray | None = None

    # ------------------------------------------------------------ geometry
    @property
    def r(self) -> np.ndarray:
        return np.sqrt(self.r_edges[1:] * self.r_edges[:-1])

    @property
    def phi(self) -> np.ndarray:
        return 0.5 * (self.phi_edges[1:] + self.phi_edges[:-1])

    @property
    def mesh(self):
        return np.meshgrid(self.r, self.phi, indexing="ij")

    def positions(self) -> np.ndarray:
        """Cartesian surface points, shape (n_r, n_phi, 3)."""
        R, P = self.mesh
        return np.stack([R * np.cos(P), R * np.sin(P), self.z], axis=-1)

    def gradients(self):
        """dz/dR and (1/R) dz/dphi (periodic in phi)."""
        R, _ = self.mesh
        dz_dr = np.gradient(self.z, self.r, axis=0)
        dphi = np.diff(self.phi_edges)[0]
        dz_dphi = (np.roll(self.z, -1, axis=1) - np.roll(self.z, 1, axis=1)) / (2.0 * dphi)
        return dz_dr, dz_dphi / R

    def normals(self) -> np.ndarray:
        """Unit normals of the upper surface, shape (n_r, n_phi, 3)."""
        _, P = self.mesh
        gr, gp = self.gradients()
        dzdx = np.cos(P) * gr - np.sin(P) * gp
        dzdy = np.sin(P) * gr + np.cos(P) * gp
        n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def areas(self) -> np.ndarray:
        """True (tilted) area of each cell, light-days^2."""
        gr, gp = self.gradients()
        dr = np.diff(self.r_edges)[:, None]
        dphi = np.diff(self.phi_edges)[None, :]
        return self.r[:, None] * dr * dphi * np.sqrt(1.0 + gr**2 + gp**2)

    # ------------------------------------------------------------- builders
    @staticmethod
    def grid(r_in: float, r_out: float, n_r: int = 300, n_phi: int = 180):
        return np.geomspace(r_in, r_out, n_r + 1), np.linspace(0.0, 2.0 * np.pi, n_phi + 1)

    @classmethod
    def from_function(cls, height: Callable, r_in: float, r_out: float, n_r: int = 300,
                      n_phi: int = 180) -> "DiscSurface":
        """``height(R, phi)`` in light-days (broadcastable arrays)."""
        r_edges, phi_edges = cls.grid(r_in, r_out, n_r, n_phi)
        R, P = np.meshgrid(np.sqrt(r_edges[1:] * r_edges[:-1]), 0.5 * (phi_edges[1:] + phi_edges[:-1]),
                           indexing="ij")
        z = np.broadcast_to(np.asarray(height(R, P), float), R.shape).copy()
        return cls(r_edges, phi_edges, z, np.ones_like(z, bool), np.zeros_like(z))

    @classmethod
    def flat(cls, r_in: float, r_out: float, **kw) -> "DiscSurface":
        return cls.from_function(lambda R, P: 0.0 * R, r_in, r_out, **kw)

    @classmethod
    def stack(cls, *surfaces: "DiscSurface") -> "DiscSurface":
        """Join surfaces in radius (same azimuthal grid, contiguous radii), e.g.
        an analytic inner disc inside a simulated outer one."""
        phi = surfaces[0].phi_edges
        for a, b in zip(surfaces[:-1], surfaces[1:]):
            if not np.allclose(a.phi_edges, b.phi_edges) or not np.isclose(a.r_edges[-1], b.r_edges[0]):
                raise ValueError("surfaces must share phi edges and be contiguous in R")
        r_edges = np.concatenate([surfaces[0].r_edges] + [s.r_edges[1:] for s in surfaces[1:]])
        cat = lambda name: np.concatenate([getattr(s, name) if getattr(s, name) is not None
                                           else np.zeros_like(s.z) for s in surfaces], axis=0)
        return cls(r_edges, phi, cat("z"), cat("valid"), cat("z_mid"))

    @classmethod
    def from_particles(cls, pos_ld: np.ndarray, r_in: float, r_out: float, n_r: int = 60, n_phi: int = 64,
                       n_sigma: float = 2.5, min_count: int = 8, center: np.ndarray | None = None,
                       smooth_phi: int = 1, smooth_r: int = 1, mass_g: np.ndarray | None = None,
                       kappa: float | None = None) -> "DiscSurface":
        """Height map from particle positions already in light-days.

        Two definitions of the surface:

        - default: ``n_sigma`` robust standard deviations above the cell's
          median height (pure geometry; ignores how much gas is there);
        - photosphere, when ``mass_g`` (particle masses in grams) and ``kappa``
          (cm^2/g, e.g. 0.4 for electron scattering) are given: each cell's gas is
          treated as a Gaussian of the measured centre and width holding the
          cell's surface density Sigma, and the surface is where the optical depth
          from above reaches 1, ``erfc(x / sqrt 2) = 2 / (kappa Sigma)``. Cells
          with ``kappa Sigma < 2`` are optically thin: no reprocessing surface
          there, so gaps and cavities open holes in the disc, and the
          photosphere rises where spiral arms pile gas up. Real AGN discs have
          ``kappa Sigma`` of 1e4-1e6, putting the photosphere at 4-5 sigma.

        Cells with fewer than ``min_count`` particles get their centre and width
        by interpolation in R along their azimuth (within the disc's radial
        extent there). ``smooth_phi`` and ``smooth_r`` cells of running-mean
        smoothing in azimuth and radius reduce particle noise, which otherwise
        casts spurious shadows: the surface sits several sigma up, so noise in
        sigma is amplified."""
        from scipy.special import erfcinv

        x = pos_ld - (np.zeros(3) if center is None else center)
        R = np.hypot(x[:, 0], x[:, 1])
        P = np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
        r_edges, phi_edges = cls.grid(r_in, r_out, n_r, n_phi)
        ir = np.searchsorted(r_edges, R) - 1
        ip = np.minimum(np.searchsorted(phi_edges, P) - 1, n_phi - 1)
        ok = (ir >= 0) & (ir < n_r)
        cell = ir[ok] * n_phi + ip[ok]
        zz = x[ok, 2]
        ncell = n_r * n_phi
        order = np.argsort(cell, kind="stable")
        cell_s, zz_s = cell[order], zz[order]
        starts = np.searchsorted(cell_s, np.arange(ncell))
        ends = np.searchsorted(cell_s, np.arange(ncell), side="right")
        zmid = np.full(ncell, np.nan)
        zsig = np.full(ncell, np.nan)
        for c in np.nonzero(ends - starts >= min_count)[0]:
            v = zz_s[starts[c]:ends[c]]
            med = np.median(v)
            zmid[c] = med
            zsig[c] = 1.4826 * np.median(np.abs(v - med))
        zmid, zsig = zmid.reshape(n_r, n_phi), zsig.reshape(n_r, n_phi)
        valid = np.isfinite(zmid)
        for arr in (zmid, zsig):
            for k in range(n_phi):
                good = np.isfinite(arr[:, k])
                if good.sum() >= 2:
                    lo, hi = np.nonzero(good)[0][[0, -1]]
                    idx = np.arange(lo, hi + 1)
                    arr[idx, k] = np.interp(idx, np.nonzero(good)[0], arr[good, k])
                    valid[idx, k] = True
        zmid = np.where(np.isfinite(zmid), zmid, 0.0)
        zsig = np.where(np.isfinite(zsig), zsig, 0.0)
        mult = np.full_like(zmid, n_sigma)
        if kappa is not None:
            if mass_g is None:
                raise ValueError("the photosphere surface needs mass_g as well as kappa")
            msum = np.bincount(cell, weights=np.broadcast_to(mass_g, R.shape)[ok], minlength=ncell)
            cell_area = (np.diff(r_edges**2)[:, None] * 0.5 * np.diff(phi_edges)[None, :]) * LIGHT_DAY_CM**2
            ksig = kappa * msum.reshape(n_r, n_phi) / cell_area
            thick = ksig > 2.0
            mult = np.where(thick, np.sqrt(2.0) * erfcinv(np.clip(2.0 / np.maximum(ksig, 1e-300), 0, 1)), 0.0)
            valid &= thick
        if smooth_phi > 1:
            kern = np.ones(smooth_phi) / smooth_phi
            pad = smooth_phi

            def _sm(a):
                ext = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
                out = np.apply_along_axis(lambda row: np.convolve(row, kern, mode="same"), 1, ext)
                return out[:, pad:-pad]
            zmid, zsig, mult = _sm(zmid), _sm(zsig), _sm(mult)
        if smooth_r > 1:
            kern_r = np.ones(smooth_r) / smooth_r

            def _smr(a):
                ext = np.pad(a, ((smooth_r, smooth_r), (0, 0)), mode="edge")
                out = np.apply_along_axis(lambda col: np.convolve(col, kern_r, mode="same"), 0, ext)
                return out[smooth_r:-smooth_r]
            zmid, zsig, mult = _smr(zmid), _smr(zsig), _smr(mult)
        z = zmid + mult * zsig
        return cls(r_edges, phi_edges, z, valid, zmid)


# ---------------------------------------------------------- analytic profiles
def flared(aspect_ratio: float, r_ref: float = 1.0, beta: float = 1.0):
    """``H = aspect_ratio * r_ref * (R / r_ref)**beta`` (beta = 9/8 is the
    classic irradiated-disc flaring of Chiang & Goldreich 1997)."""
    return lambda R, P: aspect_ratio * r_ref * (R / r_ref) ** beta


def starkey2023(height: float = 0.0, r_ref: float = 1.0, beta: float = 1.0, ripple_amplitude: float = 0.0,
                ripple_wavelength: float = 1.0, ripple_slope: float = 0.0):
    """Rim / ripple profile of Starkey, Huang, Horne & Lin (2023, eqs. 15 and 22),
    with the same parameter names as ``pycream2.rippled_disc.disc_height``."""
    def f(R, P):
        base = height * (R / r_ref) ** beta
        amp = ripple_amplitude * (R / r_ref) ** (-ripple_slope)
        return base * (1.0 + amp * np.cos(2.0 * np.pi * R / ripple_wavelength))
    return f


def spiral(amplitude: float, m: int = 2, pitch_deg: float = 15.0, base: Callable | None = None,
           r_ref: float = 1.0):
    """A logarithmic spiral corrugation of the surface (``m`` arms), e.g. the
    wake of an embedded body or a tidal perturber."""
    k = 1.0 / np.tan(np.radians(pitch_deg))
    def f(R, P):
        h0 = base(R, P) if base is not None else 0.0
        return h0 + amplitude * R / r_ref * np.cos(m * (P - k * np.log(R / r_ref)))
    return f

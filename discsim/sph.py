"""
A minimal, vectorised SPH gas disc.

Deliberately simple and readable rather than fast, so the physics is easy to
change:

- cubic-spline kernel (support 2h), smoothing length from the distance to the
  ``n_neighbours``-th neighbour (``h = d_k / 2``, so each particle has about
  that many neighbours inside its kernel);
- locally isothermal equation of state, ``P = c_s(R)^2 rho`` with
  ``c_s = aspect_ratio (R / r_ref)^flaring_index v_K(R)``, so the disc relaxes to
  ``H/R = aspect_ratio (R / r_ref)^flaring_index`` (a standard choice for
  thin-disc SPH, e.g. Lodato & Price 2010). Flaring matters for reverberation:
  a lamppost near the mid-plane only sees surfaces whose H/R grows outwards. Swap ``sound_speed`` for an irradiation-dependent one to
  make the disc's thickness respond to the lamppost (roadmap);
- Monaghan (1997) artificial viscosity with ``alpha_av``, ``beta_av``; in a disc
  it acts as a Shakura-Sunyaev viscosity of roughly
  ``alpha_SS ~ alpha_av <h> / (10 H)`` (Lodato & Price 2010), which is what
  makes the gas accrete.

Neighbour finding uses ``scipy.spatial.cKDTree``. Pairs are symmetrised so the
pressure and viscous forces conserve momentum exactly.

For production-scale runs use Phantom (Price et al. 2018), which has all of
this plus individual time steps, Lense-Thirring, sink particles and
self-gravity; ``discsim.io`` can be extended to read its dumps, and every
post-processing step (``surface``, ``irradiation``, ``reverb``) only needs
particle positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

SIGMA_3D = 1.0 / np.pi


def kernel_w(r: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Cubic spline W(r, h) in 3D (support 2h)."""
    q = r / h
    w = np.where(q < 1.0, 1.0 - 1.5 * q**2 + 0.75 * q**3,
                 np.where(q < 2.0, 0.25 * (2.0 - q) ** 3, 0.0))
    return SIGMA_3D * w / h**3


def kernel_dwdr(r: np.ndarray, h: np.ndarray) -> np.ndarray:
    """dW/dr for the cubic spline."""
    q = r / h
    d = np.where(q < 1.0, -3.0 * q + 2.25 * q**2, np.where(q < 2.0, -0.75 * (2.0 - q) ** 2, 0.0))
    return SIGMA_3D * d / h**4


@dataclass
class SPHSettings:
    aspect_ratio: float = 0.05     # H/R the isothermal disc relaxes to, at R = r_ref
    flaring_index: float = 0.0     # H/R ~ (R / r_ref)**flaring_index (2/7 for a passively irradiated disc)
    r_ref: float = 1.0
    n_neighbours: int = 40
    alpha_av: float = 0.5
    beta_av: float = 1.0
    cfl: float = 0.25


@dataclass
class SPHState:
    """Per-step quantities, kept for diagnostics and the next time step."""

    h: np.ndarray
    rho: np.ndarray
    cs: np.ndarray
    i: np.ndarray
    j: np.ndarray


class SPHGas:
    def __init__(self, settings: SPHSettings | None = None, m_bh: float = 1.0):
        self.s = settings or SPHSettings()
        self.m_bh = m_bh
        self.state: SPHState | None = None

    def sound_speed(self, pos: np.ndarray) -> np.ndarray:
        R = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) + 1e-12
        return self.s.aspect_ratio * (R / self.s.r_ref) ** self.s.flaring_index * np.sqrt(self.m_bh / R)

    def neighbours(self, pos: np.ndarray):
        """Smoothing lengths and symmetrised directed pair lists (i, j), i != j."""
        k = min(self.s.n_neighbours, len(pos) - 1)
        tree = cKDTree(pos)
        dist, idx = tree.query(pos, k=k + 1)
        h = 0.5 * dist[:, -1]
        i = np.repeat(np.arange(len(pos)), k)
        j = idx[:, 1:].ravel()
        # symmetrise: every unordered pair appears once in each direction
        key = np.concatenate([i * len(pos) + j, j * len(pos) + i])
        key = np.sort(key)
        key = key[np.concatenate([[True], key[1:] != key[:-1]])]
        return h, key // len(pos), key % len(pos)

    def density(self, pos, mass, h, i, j):
        r = np.linalg.norm(pos[i] - pos[j], axis=1)
        rho = mass * kernel_w(np.zeros_like(h), h)
        rho += np.bincount(i, weights=mass[j] * kernel_w(r, h[i]), minlength=len(pos))
        return rho

    def acceleration(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
        h, i, j = self.neighbours(pos)
        rho = self.density(pos, mass, h, i, j)
        cs = self.sound_speed(pos)
        P_over_rho2 = cs**2 / rho

        dx = pos[i] - pos[j]
        r = np.linalg.norm(dx, axis=1) + 1e-30
        dv = vel[i] - vel[j]
        hbar = 0.5 * (h[i] + h[j])
        grad = 0.5 * (kernel_dwdr(r, h[i]) + kernel_dwdr(r, h[j]))[:, None] * dx / r[:, None]

        vr = np.sum(dv * dx, axis=1)
        mu = hbar * vr / (r**2 + 0.01 * hbar**2)
        cbar = 0.5 * (cs[i] + cs[j])
        rhobar = 0.5 * (rho[i] + rho[j])
        pi_ij = np.where(vr < 0.0, (-self.s.alpha_av * cbar * mu + self.s.beta_av * mu**2) / rhobar, 0.0)

        f = (mass[j] * (P_over_rho2[i] + P_over_rho2[j] + pi_ij))[:, None] * grad
        a = -np.column_stack([np.bincount(i, weights=f[:, k], minlength=len(pos)) for k in range(3)])
        self.state = SPHState(h=h, rho=rho, cs=cs, i=i, j=j)
        return a

    def timestep(self) -> float:
        """Courant condition from the last force evaluation."""
        if self.state is None:
            return np.inf
        st = self.state
        return float(self.s.cfl * np.min(st.h / (st.cs * (1.0 + 1.2 * self.s.alpha_av))))

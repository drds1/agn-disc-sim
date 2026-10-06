"""
Gravity: the central black hole, any other massive bodies (stars, stellar-mass
black holes, a secondary SMBH, a cluster's worth of perturbers), and the
accelerations they exert on the disc's particles.

Code units throughout (``discsim.units``): G = 1, M_BH = 1.

- The central black hole is body 0. Its potential is Newtonian or
  Paczynski-Wiita (``-1 / (r - 2 r_g)``, which puts the ISCO at 6 r_g and is the
  cheapest way to get the inner disc's orbital frequencies roughly right).
- Its spin drives Lense-Thirring precession through the gravitomagnetic
  acceleration of Nelson & Papaloizou (2000, MNRAS 315, 570),
  ``a = v x H``, ``H = (2 / c^2) [J - 3 (J.r_hat) r_hat] / r^3`` with
  ``J = a_* G M^2 / c`` along the spin axis. A tilted ring's node precesses at
  ``Omega_LT = 2 G J / (c^2 r^3)``; ``tests/test_gravity.py`` checks this.
- Other massive bodies are softened point masses, integrated with each other by
  direct summation (fine for the tens to thousands of bodies of a nuclear star
  cluster's relevant members; use REBOUND if you need more, see README).
- Disc particles are massless as far as gravity goes: no self-gravity, no back
  reaction on the bodies. That is the right approximation for an AGN disc whose
  mass is a small fraction of the black hole's, at least inside the radius where
  it turns self-gravitating (a few hundred to a thousand r_g); it is listed in
  the roadmap as the first thing to relax.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MassiveBodies:
    """Point masses. Index 0 is the central black hole."""

    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    softening: np.ndarray
    names: list = field(default_factory=list)

    @classmethod
    def central(cls, mass: float = 1.0, softening: float = 0.0) -> "MassiveBodies":
        return cls(np.zeros((1, 3)), np.zeros((1, 3)), np.array([mass]), np.array([softening]), ["bh"])

    @property
    def n(self) -> int:
        return len(self.mass)

    def add(self, pos, vel, mass: float, softening: float = 0.0, name: str = "") -> int:
        self.pos = np.vstack([self.pos, np.asarray(pos, float)[None]])
        self.vel = np.vstack([self.vel, np.asarray(vel, float)[None]])
        self.mass = np.append(self.mass, mass)
        self.softening = np.append(self.softening, softening)
        self.names.append(name or f"body{self.n - 1}")
        return self.n - 1

    def add_circular(self, r: float, mass: float, inclination_deg: float = 0.0, phase_deg: float = 0.0,
                     node_deg: float = 0.0, softening: float = 0.0, name: str = "") -> int:
        """A body on a circular orbit of radius ``r`` about the central black hole
        (Keplerian speed for the central mass plus the body's own)."""
        pos, vel = circular_orbit(r, self.mass[0] + mass, inclination_deg, phase_deg, node_deg)
        return self.add(pos + self.pos[0], vel + self.vel[0], mass, softening, name)

    def copy(self) -> "MassiveBodies":
        return MassiveBodies(self.pos.copy(), self.vel.copy(), self.mass.copy(), self.softening.copy(),
                             list(self.names))


def circular_orbit(r: float, m_total: float = 1.0, inclination_deg: float = 0.0, phase_deg: float = 0.0,
                   node_deg: float = 0.0):
    """Position and velocity on a circular orbit (prograde about +z when face-on)."""
    i, u, n = np.radians([inclination_deg, phase_deg, node_deg])
    v = np.sqrt(m_total / r)
    p_orb = np.array([r * np.cos(u), r * np.sin(u), 0.0])
    v_orb = np.array([-v * np.sin(u), v * np.cos(u), 0.0])
    rot = rotation_matrix(i, n)
    return rot @ p_orb, rot @ v_orb


def rotation_matrix(inclination: float, node: float) -> np.ndarray:
    """Rotate the orbital plane by ``inclination`` about the x axis, then by the
    longitude of the ascending node ``node`` about z (radians)."""
    ci, si, cn, sn = np.cos(inclination), np.sin(inclination), np.cos(node), np.sin(node)
    rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    rz = np.array([[cn, -sn, 0], [sn, cn, 0], [0, 0, 1]])
    return rz @ rx


@dataclass
class CentralPotential:
    """Options for the central black hole (body 0)."""

    kind: str = "newton"          # "newton" or "pw" (Paczynski-Wiita)
    c: float = np.inf             # speed of light, code units (Units.c_code)
    spin: float = 0.0             # dimensionless a_*
    spin_axis: tuple = (0.0, 0.0, 1.0)

    @property
    def r_g(self) -> float:
        return 0.0 if not np.isfinite(self.c) else 1.0 / self.c**2

    @property
    def j_vec(self) -> np.ndarray:
        """BH angular momentum in code units, a_* G M^2 / c with G = M = 1."""
        if self.spin == 0.0 or not np.isfinite(self.c):
            return np.zeros(3)
        axis = np.asarray(self.spin_axis, float)
        return self.spin / self.c * axis / np.linalg.norm(axis)

    def lt_precession_rate(self, r) -> np.ndarray:
        """Nodal Lense-Thirring precession frequency 2 G J / (c^2 r^3)."""
        return 2.0 * np.linalg.norm(self.j_vec) / self.c**2 / np.asarray(r) ** 3


def central_acceleration(dx: np.ndarray, vel_rel: np.ndarray | None, m_bh: float,
                         pot: CentralPotential, softening: float = 0.0) -> np.ndarray:
    """Acceleration towards the central BH, ``dx = x - x_bh``."""
    r = np.sqrt(np.sum(dx * dx, axis=-1) + softening**2)
    if pot.kind == "pw":
        denom = np.maximum(r - 2.0 * m_bh * pot.r_g, 1e-12)
        a = -m_bh * dx / (r * denom**2)[..., None]
    elif pot.kind == "newton":
        a = -m_bh * dx / (r**3)[..., None]
    else:
        raise ValueError(f"unknown central potential {pot.kind!r}")
    j = pot.j_vec * m_bh**2
    if vel_rel is not None and np.any(j != 0.0):
        jr = dx @ j
        h = (2.0 / pot.c**2) * (j[None, :] / (r**3)[:, None] - 3.0 * (jr / r**5)[:, None] * dx)
        a = a + np.cross(vel_rel, h)
    return a


def point_mass_acceleration(dx: np.ndarray, mass: float, softening: float) -> np.ndarray:
    r2 = np.sum(dx * dx, axis=-1) + softening**2
    return -mass * dx / (r2 * np.sqrt(r2))[..., None]


def accel_on_particles(pos: np.ndarray, vel: np.ndarray, bodies: MassiveBodies,
                       pot: CentralPotential) -> np.ndarray:
    """Gravitational acceleration of every massive body on massless particles."""
    a = central_acceleration(pos - bodies.pos[0], vel - bodies.vel[0], bodies.mass[0], pot,
                             bodies.softening[0])
    for k in range(1, bodies.n):
        a += point_mass_acceleration(pos - bodies.pos[k], bodies.mass[k], bodies.softening[k])
    return a


def accel_on_bodies(bodies: MassiveBodies, pot: CentralPotential) -> np.ndarray:
    """Mutual accelerations of the massive bodies (direct summation). The
    central black hole's own potential shape (PW, spin) acts on the others;
    their pull on it is Newtonian, softened by the larger of the two lengths."""
    n = bodies.n
    a = np.zeros((n, 3))
    if n == 1:
        return a
    dx = bodies.pos[1:] - bodies.pos[0]
    a[1:] = central_acceleration(dx, bodies.vel[1:] - bodies.vel[0], bodies.mass[0], pot,
                                 bodies.softening[0])
    soft0 = np.maximum(bodies.softening[0], bodies.softening[1:])
    r2 = np.sum(dx * dx, axis=1) + soft0**2
    a[0] = np.sum(bodies.mass[1:, None] * dx / (r2 * np.sqrt(r2))[:, None], axis=0)
    for i in range(1, n):
        for j in range(i + 1, n):
            d = bodies.pos[j] - bodies.pos[i]
            s = max(bodies.softening[i], bodies.softening[j])
            r2 = d @ d + s * s
            f = d / (r2 * np.sqrt(r2))
            a[i] += bodies.mass[j] * f
            a[j] -= bodies.mass[i] * f
    return a


def total_energy(bodies: MassiveBodies, pot: CentralPotential) -> float:
    """Newtonian energy of the massive bodies (diagnostic; ignores PW/spin)."""
    ke = 0.5 * np.sum(bodies.mass * np.sum(bodies.vel**2, axis=1))
    pe = 0.0
    for i in range(bodies.n):
        for j in range(i + 1, bodies.n):
            s = max(bodies.softening[i], bodies.softening[j])
            pe -= bodies.mass[i] * bodies.mass[j] / np.sqrt(np.sum((bodies.pos[i] - bodies.pos[j]) ** 2) + s * s)
    return ke + pe

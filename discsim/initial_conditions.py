"""
Initial conditions for disc particles.

All in code units (G = M_BH = 1). Discs rotate prograde about +z unless tilted.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .gravity import rotation_matrix


def thin_disc(n: int, r_in: float = 1.0, r_out: float = 10.0, aspect_ratio: float = 0.05,
              sigma_slope: float = 1.0, disc_mass: float = 1e-3, gas: bool = True,
              tilt_deg: float | Callable = 0.0, twist_deg: float | Callable = 0.0,
              flaring_index: float = 0.0, r_ref: float = 1.0, seed: int = 0):
    """Particles of a thin disc with ``Sigma ~ R**-sigma_slope`` from ``r_in`` to
    ``r_out`` and a Gaussian vertical profile of scale height ``aspect_ratio * R``.

    With ``flaring_index`` f, the scale height is ``aspect_ratio (R/r_ref)^f R``
    (match ``SPHSettings``).

    ``gas=True`` subtracts the radial pressure support of a locally isothermal
    disc from the orbital speed (``v_phi^2 = v_K^2 [1 - (p + 2 - f) (H/R)^2]``), so
    the SPH disc starts close to equilibrium; ``gas=False`` gives test particles
    exactly Keplerian circular velocities (their vertical spread then just
    becomes small orbital inclinations).

    ``tilt_deg`` and ``twist_deg`` may be numbers or functions of R (in degrees),
    giving a warped disc: each annulus is rotated by its tilt about the x axis,
    then by its twist about z.

    Returns ``pos, vel, mass``.
    """
    rng = np.random.default_rng(seed)
    p = sigma_slope
    # sample R from dN ~ Sigma R dR ~ R^(1-p) dR
    u = rng.uniform(size=n)
    if np.isclose(p, 2.0):
        R = r_in * (r_out / r_in) ** u
    else:
        e = 2.0 - p
        R = (r_in**e + u * (r_out**e - r_in**e)) ** (1.0 / e)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    hr = aspect_ratio * (R / r_ref) ** flaring_index
    z = rng.normal(0.0, hr * R)
    vk = np.sqrt(1.0 / R)
    vphi = vk * np.sqrt(np.clip(1.0 - (p + 2.0 - flaring_index) * hr**2, 0.0, None)) if gas else vk
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), z])
    vel = np.column_stack([-vphi * np.sin(phi), vphi * np.cos(phi), np.zeros(n)])
    if not gas:
        # a test particle at height z on a circular orbit of the same R: give it the
        # vertical velocity of an orbit inclined by z/R, with a random phase, so the
        # ensemble is stationary rather than collapsing to the mid-plane together
        incl = z / R
        psi = rng.uniform(0.0, 2.0 * np.pi, n)
        pos[:, 2] = R * incl * np.sin(psi)
        vel[:, 2] = vk * incl * np.cos(psi)

    tilt = tilt_deg(R) if callable(tilt_deg) else np.full(n, float(tilt_deg))
    twist = twist_deg(R) if callable(twist_deg) else np.full(n, float(twist_deg))
    if np.any(tilt != 0) or np.any(twist != 0):
        rots = np.stack([rotation_matrix(a, b) for a, b in zip(np.radians(tilt), np.radians(twist))])
        pos = np.einsum("nij,nj->ni", rots, pos)
        vel = np.einsum("nij,nj->ni", rots, vel)
    mass = np.full(n, disc_mass / n)
    return pos, vel, mass


def ring(n: int, r: float, inclination_deg: float = 0.0, node_deg: float = 0.0, seed: int = 0):
    """``n`` test particles on one circular orbit (for precession tests)."""
    rng = np.random.default_rng(seed)
    u = np.sort(rng.uniform(0, 2 * np.pi, n))
    rot = rotation_matrix(np.radians(inclination_deg), np.radians(node_deg))
    v = np.sqrt(1.0 / r)
    pos = np.column_stack([r * np.cos(u), r * np.sin(u), np.zeros(n)]) @ rot.T
    vel = np.column_stack([-v * np.sin(u), v * np.cos(u), np.zeros(n)]) @ rot.T
    return pos, vel, np.full(n, 1e-12)


def nuclear_star_cluster(n: int, r_scale: float = 50.0, cluster_mass: float = 0.1, m_star: float | None = None,
                         r_max: float | None = None, seed: int = 0):
    """An isotropic Plummer sphere of ``n`` perturbers around the central BH,
    velocities drawn so each is bound to the BH-plus-cluster potential.

    Returns ``pos, vel, masses``; add them with ``MassiveBodies.add``.
    A real nuclear star cluster has ~10^7 stars; only the handful that cross
    the disc each orbit matter for its structure, so a few hundred heavy
    "super-stars" are a reasonable first experiment (and n^2 direct summation
    stays cheap)."""
    rng = np.random.default_rng(seed)
    m = cluster_mass / n if m_star is None else m_star
    u = rng.uniform(1e-3, 0.999, n)
    r = r_scale / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    if r_max is not None:
        r = np.minimum(r, r_max)
    cth = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    sth = np.sqrt(1 - cth**2)
    pos = r[:, None] * np.column_stack([sth * np.cos(ph), sth * np.sin(ph), cth])
    # isotropic speed below escape: v = f v_esc with f from the Plummer distribution, simplified:
    m_enc = 1.0 + cluster_mass * r**3 / (r**2 + r_scale**2) ** 1.5
    v_circ = np.sqrt(m_enc / r)
    v = v_circ * rng.uniform(0.5, 1.2, n)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return pos, v[:, None] * d, np.full(n, m)

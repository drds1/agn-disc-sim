"""
Snapshot -> surface -> irradiated disc -> response functions, in one call.
"""

from __future__ import annotations

import numpy as np

from .reverb import DiscPhysics, Observer, ResponseSet, irradiate, response_functions
from .simulation import Snapshot
from .surface import DiscSurface
from .units import MSUN_G, Units


def snapshot_surface(snap: Snapshot, units: Units, r_in_ld: float, r_out_ld: float, **surface_kw) -> DiscSurface:
    """Surface map of a snapshot (centred on the central black hole). Pass
    ``kappa`` (cm^2/g) in ``surface_kw`` for the photosphere surface; particle
    masses are converted to grams from ``units``."""
    pos_ld = (snap.pos - snap.body_pos[0]) * units.length_ld
    if surface_kw.get("kappa") is not None and surface_kw.get("mass_g") is None:
        surface_kw = dict(surface_kw, mass_g=snap.mass * units.m_bh * MSUN_G)
    return DiscSurface.from_particles(pos_ld, r_in_ld, r_out_ld, **surface_kw)


def snapshot_responses(snap: Snapshot, units: Units, physics: DiscPhysics, observer: Observer, wavelengths,
                       r_in_ld: float, r_out_ld: float, surface_kw: dict | None = None,
                       response_kw: dict | None = None):
    """Returns ``(surface, irradiated_disc, ResponseSet)``.

    ``physics.m_bh`` should match ``units.m_bh``: the dynamics are scale-free,
    the radiation is not."""
    if not np.isclose(physics.m_bh, units.m_bh):
        raise ValueError("physics.m_bh and units.m_bh differ")
    surf = snapshot_surface(snap, units, r_in_ld, r_out_ld, **(surface_kw or {}))
    disc = irradiate(surf, physics)
    rs: ResponseSet = response_functions(disc, observer, wavelengths, **(response_kw or {}))
    return surf, disc, rs

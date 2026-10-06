"""
Unit system.

Code units: G = 1, M_BH = 1, length unit = ``length_ld`` light-days (default 1).
The time unit follows, ``T = sqrt(L**3 / (G M))``. Reverberation post-processing
(``discsim.irradiation``, ``discsim.reverb``) works in light-days and days, so a
``Units`` object is what converts a dynamical snapshot into something with a
temperature and a delay.

For M_BH = 1e8 Msun and L = 1 light-day: T = 13.2 days, so an orbit at 1 light-day
takes 83 days, the speed of light is 13.2 code units and r_g = GM/c^2 = 0.0057
light-days. Orbits are therefore far slower than light-travel delays everywhere
outside a few r_g: a dynamical snapshot is effectively frozen during a reverberation
signal, which is why the reverberation layer works on static snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# cgs
G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
MSUN_G = 1.98847e33
SIGMA_SB = 5.670374e-5
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
DAY_S = 86400.0
YEAR_S = 3.15576e7
LIGHT_DAY_CM = C_CGS * DAY_S
MPC_CM = 3.0856776e24
THOMSON_CM2 = 6.6524587e-25
PROTON_G = 1.67262192e-24


@dataclass(frozen=True)
class Units:
    """Code units for a black hole of mass ``m_bh`` (solar masses)."""

    m_bh: float = 1.0e8
    length_ld: float = 1.0

    @property
    def gm_cgs(self) -> float:
        return G_CGS * self.m_bh * MSUN_G

    @property
    def length_cm(self) -> float:
        return self.length_ld * LIGHT_DAY_CM

    @property
    def time_s(self) -> float:
        return float(np.sqrt(self.length_cm**3 / self.gm_cgs))

    @property
    def time_days(self) -> float:
        return self.time_s / DAY_S

    @property
    def velocity_cms(self) -> float:
        return self.length_cm / self.time_s

    @property
    def c_code(self) -> float:
        """Speed of light in code units."""
        return C_CGS / self.velocity_cms

    @property
    def r_g(self) -> float:
        """Gravitational radius GM/c^2 in code length units."""
        return 1.0 / self.c_code**2

    @property
    def r_g_ld(self) -> float:
        return self.r_g * self.length_ld

    @property
    def l_edd_cgs(self) -> float:
        return 4.0 * np.pi * self.gm_cgs * C_CGS * PROTON_G / THOMSON_CM2

    def orbital_period_days(self, r_code) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(r_code) ** 1.5 * self.time_days

    def mdot_edd_msun_yr(self, efficiency: float = 0.1) -> float:
        return self.l_edd_cgs / (efficiency * C_CGS**2) * YEAR_S / MSUN_G

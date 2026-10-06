"""
The simulation driver: massive bodies + disc particles (test particles or SPH
gas), advanced with a kick-drift-kick leapfrog on a shared time step.

Time step: the smaller of
- ``eta_orbit`` times the shortest orbital time ``sqrt(r^3 / M)`` among the
  particles and bodies (measured from the central BH),
- ``eta_orbit`` times the shortest close-encounter time between a body and
  the central BH or another body,
- the SPH Courant step, if gas is present.

Velocity-dependent forces (Lense-Thirring, artificial viscosity) are evaluated
with the start-of-kick velocity: first-order in that small term, which is the
usual compromise in SPH codes.

Particles that fall inside ``r_accrete`` of the central BH, or within an embedded
body's ``softening``-sized accretion radius if ``body_accretion`` is on, are
removed and their mass added to an accreted-mass tally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gravity import CentralPotential, MassiveBodies, accel_on_bodies, accel_on_particles
from .sph import SPHGas


@dataclass
class Snapshot:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    ids: np.ndarray
    body_pos: np.ndarray
    body_vel: np.ndarray
    body_mass: np.ndarray
    h: np.ndarray | None = None
    rho: np.ndarray | None = None

    def save(self, path) -> None:
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items() if v is not None})

    @classmethod
    def load(cls, path) -> "Snapshot":
        d = np.load(path)
        return cls(**{k: (d[k] if k in d else None) for k in cls.__dataclass_fields__})


@dataclass
class Simulation:
    bodies: MassiveBodies
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    potential: CentralPotential = field(default_factory=CentralPotential)
    gas: SPHGas | None = None
    r_accrete: float = 0.0
    body_accretion: bool = False
    eta_orbit: float = 0.02
    dt_max: float = np.inf
    t: float = 0.0
    accreted_mass: float = 0.0
    ids: np.ndarray | None = None

    def __post_init__(self):
        if self.ids is None:
            self.ids = np.arange(len(self.pos))
        self._acc_p = None
        self._acc_b = None

    # ---------------------------------------------------------------- forces
    def _accelerations(self):
        a_p = accel_on_particles(self.pos, self.vel, self.bodies, self.potential)
        if self.gas is not None and len(self.pos) > 1:
            a_p = a_p + self.gas.acceleration(self.pos, self.vel, self.mass)
        a_b = accel_on_bodies(self.bodies, self.potential)
        return a_p, a_b

    def timestep(self) -> float:
        dt = self.dt_max
        x0 = self.bodies.pos[0]
        if len(self.pos):
            r = np.linalg.norm(self.pos - x0, axis=1)
            dt = min(dt, self.eta_orbit * np.sqrt(np.min(r) ** 3 / self.bodies.mass[0]))
        b = self.bodies
        for k in range(1, b.n):
            for m in range(k):
                d = np.linalg.norm(b.pos[k] - b.pos[m]) + max(b.softening[k], b.softening[m])
                dt = min(dt, self.eta_orbit * np.sqrt(d**3 / (b.mass[k] + b.mass[m])))
        if self.gas is not None:
            dt = min(dt, self.gas.timestep())
        return float(dt)

    # ------------------------------------------------------------- stepping
    def _remove_accreted(self):
        if not len(self.pos):
            return
        keep = np.ones(len(self.pos), bool)
        if self.r_accrete > 0:
            keep &= np.linalg.norm(self.pos - self.bodies.pos[0], axis=1) > self.r_accrete
        if self.body_accretion:
            for k in range(1, self.bodies.n):
                if self.bodies.softening[k] > 0:
                    keep &= np.linalg.norm(self.pos - self.bodies.pos[k], axis=1) > self.bodies.softening[k]
        if not keep.all():
            self.accreted_mass += self.mass[~keep].sum()
            self.pos, self.vel, self.mass, self.ids = (self.pos[keep], self.vel[keep], self.mass[keep],
                                                      self.ids[keep])
            self._acc_p = None

    def step(self, dt: float | None = None) -> float:
        if self._acc_p is None:
            self._acc_p, self._acc_b = self._accelerations()
        if dt is None:
            dt = self.timestep()
        b = self.bodies
        self.vel += 0.5 * dt * self._acc_p
        b.vel += 0.5 * dt * self._acc_b
        self.pos += dt * self.vel
        b.pos += dt * b.vel
        self._remove_accreted()
        self._acc_p, self._acc_b = self._accelerations()
        self.vel += 0.5 * dt * self._acc_p
        b.vel += 0.5 * dt * self._acc_b
        self.t += dt
        return dt

    def run(self, t_end: float, snapshot_every: float | None = None, outdir: str | Path | None = None,
            max_steps: int = 10_000_000, verbose: bool = True) -> list[Snapshot]:
        """Advance to ``t_end``. Returns snapshots taken every ``snapshot_every``
        (and at the end); saves them to ``outdir`` as ``snap_XXXX.npz`` if given."""
        snaps = []
        next_snap = self.t if snapshot_every else np.inf
        if outdir is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
        clock, n = time.time(), 0
        while self.t < t_end and n < max_steps:
            if self.t >= next_snap - 1e-12:
                snaps.append(self.snapshot())
                if outdir is not None:
                    snaps[-1].save(Path(outdir) / f"snap_{len(snaps) - 1:04d}.npz")
                next_snap += snapshot_every
            dt = min(self.timestep(), t_end - self.t)
            if snapshot_every:
                dt = min(dt, max(next_snap - self.t, 1e-12))
            self.step(dt)
            n += 1
            if verbose and n % 200 == 0:
                print(f"  step {n:6d}  t = {self.t:9.4f}  dt = {dt:.2e}  N = {len(self.pos)}  "
                      f"wall = {time.time() - clock:6.1f} s", flush=True)
        snaps.append(self.snapshot())
        if outdir is not None:
            snaps[-1].save(Path(outdir) / f"snap_{len(snaps) - 1:04d}.npz")
        return snaps

    def snapshot(self) -> Snapshot:
        st = self.gas.state if self.gas is not None else None
        same = st is not None and len(st.h) == len(self.pos)
        return Snapshot(self.t, self.pos.copy(), self.vel.copy(), self.mass.copy(), self.ids.copy(),
                        self.bodies.pos.copy(), self.bodies.vel.copy(), self.bodies.mass.copy(),
                        st.h.copy() if same else None, st.rho.copy() if same else None)

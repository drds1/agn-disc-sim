import numpy as np

from discsim import CentralPotential, MassiveBodies, Simulation, Units, ring
from discsim.gravity import circular_orbit, total_energy


def _node(pos, vel):
    L = np.cross(pos, vel).mean(axis=0)
    return np.arctan2(L[0], -L[1])


def test_units_for_1e8_msun():
    u = Units(1e8)
    assert np.isclose(u.r_g_ld, 0.00571, rtol=0.01)
    assert np.isclose(u.time_days, 13.2, rtol=0.01)
    assert np.isclose(u.orbital_period_days(1.0), 83.0, rtol=0.01)


def test_two_body_energy_conserved_over_20_orbits():
    b = MassiveBodies.central()
    b.add_circular(3.0, 1e-3)
    e0 = total_energy(b, CentralPotential())
    sim = Simulation(b, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), eta_orbit=0.01)
    sim.run(20 * 2 * np.pi * 3**1.5, verbose=False)
    assert abs(total_energy(sim.bodies, CentralPotential()) / e0 - 1) < 1e-6


def test_test_particle_circular_orbit_stays_circular():
    pos, vel = circular_orbit(2.0)
    sim = Simulation(MassiveBodies.central(), pos[None], vel[None], np.ones(1), eta_orbit=0.01)
    sim.run(10 * 2 * np.pi * 2**1.5, verbose=False)
    assert abs(np.linalg.norm(sim.pos[0]) - 2.0) < 1e-3


def test_lense_thirring_nodal_precession_rate():
    pot = CentralPotential(c=5.0, spin=0.9)
    pos, vel, m = ring(4, 2.0, inclination_deg=10.0)
    sim = Simulation(MassiveBodies.central(), pos, vel, m, potential=pot, eta_orbit=0.005)
    n0 = _node(sim.pos, sim.vel)
    sim.run(200.0, verbose=False)
    rate = (_node(sim.pos, sim.vel) - n0) / 200.0
    assert np.isclose(rate, pot.lt_precession_rate(2.0), rtol=0.03)  # prograde, at the analytic rate


def test_paczynski_wiita_is_stronger_than_newton_near_the_hole():
    from discsim.gravity import central_acceleration
    dx = np.array([[10.0 / 13.2**2, 0, 0]])  # 10 r_g for c = 13.2
    pw = central_acceleration(dx, None, 1.0, CentralPotential("pw", c=13.2))
    nw = central_acceleration(dx, None, 1.0, CentralPotential("newton", c=13.2))
    assert np.isclose(pw[0, 0] / nw[0, 0], (10 / 8) ** 2, rtol=1e-6)

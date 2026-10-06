import numpy as np

from discsim import drw_driver, echo
from discsim.reverb import ResponseSet, Observer


def test_echo_of_a_delta_response_is_a_shifted_driver():
    t, x = drw_driver(300.0, dt=0.1, tau_drw=30.0, seed=1)
    tau = np.linspace(0.05, 20.0, 200)
    psi = np.exp(-0.5 * ((tau - 5.0) / 0.1) ** 2)
    psi /= np.trapezoid(psi, tau)
    y = echo(t, x, ResponseSet(tau, np.array([5000.0]), psi[None], Observer()))[0]
    shift = int(round(5.0 / 0.1))
    assert np.corrcoef(y[shift + 300:], x[300:-shift])[0, 1] > 0.999


def test_drw_variance():
    _, x = drw_driver(20000.0, dt=1.0, tau_drw=10.0, sigma=0.2, seed=0)
    assert np.isclose(np.std(x), 0.2, rtol=0.1)

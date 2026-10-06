"""discsim: N-body / SPH simulations of AGN accretion discs and their
continuum reverberation signatures.

Layers (each usable on its own):

- dynamics: ``gravity``, ``sph``, ``initial_conditions``, ``simulation``
- geometry: ``surface`` (particles or analytic profiles -> reprocessing surface)
- radiation: ``reverb`` (lamppost irradiation, shadows, response functions, SED)
- observables: ``lightcurves`` (mock campaigns in pycream2's input format)
"""

from .gravity import CentralPotential, MassiveBodies
from .initial_conditions import nuclear_star_cluster, ring, thin_disc
from .lightcurves import MockCampaign, drw_driver, echo, write_pycream2
from .reverb import DiscPhysics, Observer, irradiate, response_functions, sed, thin_disc_lag
from .simulation import Simulation, Snapshot
from .sph import SPHGas, SPHSettings
from .surface import DiscSurface, flared, spiral, starkey2023
from .units import Units

__version__ = "0.1.0"

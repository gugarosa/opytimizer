"""An evolutionary package for all common opytimizer modules.
It contains implementations of swarm-based optimizers.
"""

from opytimizer.optimizers.swarm.abc import ABC
from opytimizer.optimizers.swarm.abo import ABO
from opytimizer.optimizers.swarm.af import AF
from opytimizer.optimizers.swarm.ba import BA
from opytimizer.optimizers.swarm.boa import BOA
from opytimizer.optimizers.swarm.bwo import BWO
from opytimizer.optimizers.swarm.cs import CS
from opytimizer.optimizers.swarm.csa import CSA
from opytimizer.optimizers.swarm.eho import EHO
from opytimizer.optimizers.swarm.fa import FA
from opytimizer.optimizers.swarm.ffoa import FFOA
from opytimizer.optimizers.swarm.fpa import FPA
from opytimizer.optimizers.swarm.fso import FSO
from opytimizer.optimizers.swarm.goa import GOA
from opytimizer.optimizers.swarm.js import JS, NBJS
from opytimizer.optimizers.swarm.kh import KH
from opytimizer.optimizers.swarm.mfo import MFO
from opytimizer.optimizers.swarm.mrfo import MRFO
from opytimizer.optimizers.swarm.pio import PIO
from opytimizer.optimizers.swarm.pso import AIWPSO, PSO, RPSO, SAVPSO, VPSO
from opytimizer.optimizers.swarm.sbo import SBO
from opytimizer.optimizers.swarm.sca import SCA
from opytimizer.optimizers.swarm.sfo import SFO
from opytimizer.optimizers.swarm.sos import SOS
from opytimizer.optimizers.swarm.ssa import SSA
from opytimizer.optimizers.swarm.sso import SSO
from opytimizer.optimizers.swarm.stoa import STOA
from opytimizer.optimizers.swarm.woa import WOA

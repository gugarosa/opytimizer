# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Hill-Climbing.

Updates perturb all agent positions with Gaussian noise, following page 252 of the reference.

References:
    S. Skiena. The Algorithm Design Manual (2010).

"""

from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class HC(Optimizer):
    """Perturb a population with Gaussian hill-climbing steps.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the Gaussian perturbation distribution.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``r_mean`` (noise mean, 0.0) and
            ``r_var`` (noise standard deviation passed to NumPy, 0.1).

        """

        super(HC, self).__init__()

        self.r_mean = 0.0
        self.r_var = 0.1

        self.build(params)

    def update(self, space: Space) -> None:
        for agent in space.agents:
            noise = np.random.normal(self.r_mean, self.r_var, (agent.n_variables, agent.n_dimensions))
            agent.position += noise

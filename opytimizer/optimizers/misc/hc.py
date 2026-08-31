"""Hill-Climbing."""

from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class HC(Optimizer):
    """An HC class, inherited from Optimizer.

    This is the designed class to define HC-related
    variables and methods.

    References:
        S. Skiena. The Algorithm Design Manual (2010).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(HC, self).__init__()

        self.r_mean = 0.0
        self.r_var = 0.1

        self.build(params)

    def update(self, space: Space) -> None:
        """Wraps Hill Climbing over all agents and variables (p. 252).

        Args:
            space: Space containing agents and update-related information.

        """

        for agent in space.agents:
            noise = np.random.normal(
                self.r_mean, self.r_var, (agent.n_variables, agent.n_dimensions)
            )
            agent.position += noise

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Provide shared configuration and evaluation hooks for optimization strategies.

The base optimizer remains concrete for evaluation-only workflows.
Subclasses override compilation or movement only when those responsibilities differ.

"""

import copy
import time
from collections.abc import Callable, Mapping
from typing import Any

from opytimizer.core.space import Space


class Optimizer:
    """Provide the strategy hooks used by :class:`opytimizer.Opytimizer`.

    """

    def build(self, params: Mapping[str, Any] | None = None) -> None:
        """Apply parameter overrides to this optimizer without copying values.

        Args:
            params: Attribute overrides.

        Raises:
            TypeError: The overrides are not a mapping or None.

        Notes:
            None leaves existing values intact. Keys are not restricted and values are not copied.
            Subclasses validate meaningful parameter domains at configuration or consumption boundaries.

        """

        if params is None:
            return
        if not isinstance(params, Mapping):
            raise TypeError("`params` must be a mapping.")

        for key, value in params.items():
            setattr(self, key, value)

    def compile(self, space: Space) -> None:
        """Prepare space-dependent state before the first evaluation.

        ``Opytimizer`` calls this once during construction, not on every
        ``start``. Override it to allocate buffers using the space's population
        and position dimensions. The default implementation does nothing.

        Args:
            space: Initialized population whose dimensions determine state shape.

        """

        pass

    def evaluate(self, space: Space, function: Callable) -> None:
        """Evaluate current positions and retain strict global improvements.

        Args:
            space: Population whose agents and best-agent state are updated.
            function: Scalar objective receiving a live array of shape ``(n_variables, n_dimensions)``.

        Notes:
            Exceptions from the objective propagate. Subclasses may override
            this hook for different state semantics, such as PSO personal bests.

        """

        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self) -> None:
        """Move candidates in a subclass-specific way.

        The default implementation does nothing. Override this method with named
        positional arguments matching ``Opytimizer`` attributes, commonly
        ``space``, ``function``, ``iteration``, and ``n_iterations``. The driver
        resolves those arguments by name and clips the population afterward.

        """

        pass

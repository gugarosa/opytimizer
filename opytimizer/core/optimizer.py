"""Shared configuration and evaluation hooks for optimization strategies."""

import copy
import time
from collections.abc import Callable, Mapping
from typing import Any

from opytimizer.core.space import Space


class Optimizer:
    """Define the strategy hooks used by :class:`opytimizer.Opytimizer`.

    Override ``update`` to move candidates and ``compile`` when the algorithm
    needs space-dependent state. The default evaluator minimizes a scalar
    objective and maintains an independent best-position snapshot.

    The base class is intentionally instantiable: its compilation and update
    hooks do nothing, which is useful for evaluation-only workflows.
    """

    def build(self, params: Mapping[str, Any] | None = None) -> None:
        """Apply parameter overrides to this optimizer without copying values.

        Args:
            params: Attribute names and values. ``None`` leaves defaults intact.

        Raises:
            TypeError: If ``params`` is not a mapping or ``None``.

        Notes:
            This method does not restrict attribute names or validate algorithm
            domains. Subclasses own those checks; use their documented parameter
            names rather than relying on misspellings to be rejected.
        """

        if params is None:
            return
        if not isinstance(params, Mapping):
            raise TypeError("`params` should be a mapping or None")

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
            function: Callable receiving a live position array of shape
                ``(n_variables, n_dimensions)`` and returning a scalar fitness.

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

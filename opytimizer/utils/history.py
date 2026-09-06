"""Snapshots and convergence views of optimization history."""

from typing import Any

import numpy as np


class History:
    """Append named observations and retrieve their concatenated history.

    ``dump`` creates history attributes lazily. Agent positions are copied into
    Python lists; arbitrary values are retained as supplied. The driver records
    agent/best-agent snapshots per completed iteration and elapsed ``time`` per
    normally completed run.
    """

    def __init__(self, save_agents: bool = False) -> None:
        """Configure whether population snapshots should be retained.

        Args:
            save_agents: Retain ``agents`` records as well as best-agent records.
                Population histories can be much larger than best-only histories.

        """

        if not isinstance(save_agents, bool):
            raise TypeError("`save_agents` should be a boolean")

        self.save_agents = save_agents

    def _parse(
        self, key: str, value: Any
    ) -> list[Any] | tuple[list[Any], float] | None:
        """Copy position arrays into the known history record formats."""

        if key == "agents":
            return [(v.position.tolist(), v.fit) for v in value]

        if key == "best_agent":
            return (value.position.tolist(), value.fit)

        if key == "local_position":
            return [v.tolist() for v in value]

    def dump(self, **kwargs: Any) -> None:
        """Append one observation per named history.

        ``agents`` stores a population of position/fitness pairs; ``best_agent``
        stores one such pair; ``local_position`` stores position lists. Other
        values are appended without copying. ``agents`` is omitted entirely when
        ``save_agents`` is false.
        """

        for key, value in kwargs.items():
            if key == "agents" and not self.save_agents:
                continue

            if key in ("agents", "best_agent", "local_position"):
                output = self._parse(key, value)
            else:
                output = value

            if not hasattr(self, key):
                setattr(self, key, [output])
            else:
                getattr(self, key).append(output)

    def get_convergence(
        self, key: str, index: int | tuple[int, ...] | None = 0
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Concatenate recorded observations without changing stored snapshots.

        Args:
            key: Name previously recorded by ``dump``.
            index: NumPy index selecting an agent for ``agents`` or an entry in
                ``local_position``. Ignored for other keys.

        Returns:
            A ``(positions, fitness)`` pair for ``agents`` and ``best_agent``.
            With an integer agent index and ``(n_variables, n_dimensions)``
            positions, these have shapes
            ``(n_variables, n_records * n_dimensions)`` and ``(n_records,)``.
            Other keys return one array concatenated with ``numpy.hstack``.

        Raises:
            AttributeError: If the key has not been recorded, including disabled
                population histories.
        """

        attr = np.asarray(getattr(self, key), dtype=object)

        if key == "agents":
            attr_pos = np.hstack(attr[:, index, 0])
            attr_fit = np.hstack(attr[:, index, 1])

            return attr_pos, attr_fit

        if key == "best_agent":
            attr_pos = np.hstack(attr[:, 0])
            attr_fit = np.hstack(attr[:, 1])

            return attr_pos, attr_fit

        if key == "local_position":
            return np.hstack(attr[:, index])

        return np.hstack(attr[:])

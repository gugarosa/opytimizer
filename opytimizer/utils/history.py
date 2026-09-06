# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Provide snapshots and convergence views of optimization history.

"""

from typing import Any

import numpy as np


class History:
    """Record optimization snapshots and retrieve their concatenated history.

    """

    def __init__(self, save_agents: bool = False) -> None:
        """Configure whether population snapshots should be retained.

        Histories are created lazily, one list per key. Agent positions are copied into lists.
        Arbitrary custom values remain live references. The driver records snapshots per completed iteration
        and elapsed time per normally completed run.

        Args:
            save_agents: Whether to retain population records in addition to best-agent records.

        """

        if not isinstance(save_agents, bool):
            raise TypeError("`save_agents` must be a boolean.")

        self.save_agents = save_agents

    def dump(self, **kwargs: Any) -> None:
        """Append one observation per named history.

        ``agents`` stores population position/fitness pairs. ``best_agent`` stores one pair.
        ``local_position`` stores position lists. Other values are appended without copying.
        Population records are omitted when ``save_agents`` is False.

        Args:
            **kwargs: Named observations to append.

        """

        for key, value in kwargs.items():
            if key == "agents":
                if not self.save_agents:
                    continue
                output = [(agent.position.tolist(), agent.fit) for agent in value]
            elif key == "best_agent":
                output = (value.position.tolist(), value.fit)
            elif key == "local_position":
                output = [position.tolist() for position in value]
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
            index: Agent or local-position index, ignored for other keys.

        Returns:
            A position/fitness pair for agent records, or a horizontally concatenated array for other records.

        Raises:
            AttributeError: The key has not been recorded.

        Notes:
            Integer-indexed position histories have shape ``(n_variables, n_records * n_dimensions)``.
            Their fitness histories have shape ``(n_records,)``.

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

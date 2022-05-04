"""History-based object that helps in saving the optimization history.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

import opytimizer.utils.exception as e


class History:
    """A History class is responsible for saving each iteration's output.

    Note that you can use dump() and parse() for whatever your needs. Our default
    is only for agents, best agent and best agent's index.

    """

    def __init__(self, save_agents: Optional[bool] = False) -> None:
        """Initialization method.

        Args:
            save_agents: Saves all agents in the search space.

        """

        # Stores only the best agent
        self.save_agents = save_agents

    @property
    def save_agents(self) -> bool:
        """Saves all agents in the search space."""

        return self._save_agents

    @save_agents.setter
    def save_agents(self, save_agents: bool) -> None:
        if not isinstance(save_agents, bool):
            raise e.TypeError("`save_agents` should be a boolean")

        self._save_agents = save_agents

    def _parse(self, key: str, value: Any) -> Union[List[Any], Tuple[List[Any], float]]:
        """Parses incoming values with specified formats.

        Args:
            key: Key.
            value: Value.

        Returns:
            (Union[List[Any], Tuple[List[Any], float]]): Parsed value according to the specified format.

        """

        if key == "agents":
            # Returns a list of tuples (position, fit)
            return [(v.position.tolist(), v.fit) for v in value]

        if key == "best_agent":
            # Returns a tuple (position, fit)
            return (value.position.tolist(), value.fit)

        if key == "local_position":
            # Returns a list of local positions
            return [v.tolist() for v in value]

    def dump(self, **kwargs) -> None:
        """Dumps keyword pairs into self-class attributes."""

        for (key, value) in kwargs.items():
            # If current `key` is `agents` and they should not be saved,
            # we skip this loop iteration
            if key == "agents" and not self.save_agents:
                continue

            # If current `key` has a specific parsing rule,
            # we need to parse it accordingly
            if key in ["agents", "best_agent", "local_position"]:
                output = self._parse(key, value)
            else:
                output = value

            # If class still does not have a `key` property,
            # we need to set its initial value as a list
            if not hasattr(self, key):
                setattr(self, key, [output])
            else:
                getattr(self, key).append(output)

    def get_convergence(
        self, key: str, index: Optional[Tuple[int, ...]] = 0
    ) -> np.ndarray:
        """Gets the convergence list of a specified key.

        Args:
            key: Key to be retrieved.
            index: Index to be retrieved.

        Returns:
            (np.ndarray): Values based on key and index.

        """

        # Gathers the numpy array from the attribute
        attr = np.asarray(getattr(self, key), dtype=list)

        if key in ["agents"]:
            # Gathers positions and fitnesses
            attr_pos = np.hstack(attr[(slice(None), index, 0)])
            attr_fit = np.hstack(attr[(slice(None), index, 1)])

            return attr_pos, attr_fit

        if key in ["best_agent"]:
            # Gathers positions and fitnesses
            attr_pos = np.hstack(attr[(slice(None), 0)])
            attr_fit = np.hstack(attr[(slice(None), 1)])

            return attr_pos, attr_fit

        if key in ["local_position"]:
            # Gathers positions
            attr_pos = np.hstack(attr[(slice(None), index)])

            return attr_pos

        # Gathers the attribute
        attr = np.hstack(attr[(slice(None))])

        return attr

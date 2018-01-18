import numpy as np
import pytest

from opytimizer.core import agent


def test_agent_creation():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)
    assert new_agent.position.shape == (5, 4)

import os

import pytest

from opytimizer.core import agent
from opytimizer.utils import history


def test_history_dump():
    new_history = history.History()

    agents = [agent.Agent(n_variables=2, n_dimensions=1) for _ in range(5)]

    new_history.dump(agents=agents, best_agent=agents[4], value=0)

    assert len(new_history.agents) > 0
    assert len(new_history.best_agent) > 0
    assert new_history.value[0] == 0


def test_history_get():
    new_history = history.History()

    agents = [agent.Agent(n_variables=2, n_dimensions=1) for _ in range(5)]

    new_history.dump(agents=agents, best_agent=agents[4], value=0)

    try:
        agents = new_history.get(key='agents', index=0)
    except:
        agents = new_history.get(key='agents', index=(0, 0))

    try:
        agents = new_history.get(key='agents', index=(0, 0, 0))
    except:
        agents = new_history.get(key='agents', index=(0, 0))

    assert agents.shape == (2, 1)


def test_history_save():
    new_history = history.History()

    agents = [agent.Agent(n_variables=2, n_dimensions=1) for _ in range(5)]

    new_history.dump(agents=agents, best_agent=agents[0])

    new_history.save('models/test.pkl')

    assert os.path.isfile('./models/test.pkl')


def test_history_load():
    new_history = history.History()

    new_history.load('models/test.pkl')

    assert len(new_history.agents) > 0

    print(new_history)

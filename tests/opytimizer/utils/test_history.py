import os

import pytest
from opytimizer.core import agent
from opytimizer.utils import history


def test_history_history():
    new_history = history.History()

    assert type(new_history.history) == list


def test_history_history_setter():
    new_history = history.History()

    new_history.history = [1, 2, 3]

    assert len(new_history.history) > 0


def test_history_dump():
    new_history = history.History()

    agents = []
    for _ in range(5):
        agents.append(agent.Agent(n_variables=2, n_dimensions=1))

    new_history.dump(agents)

    assert len(new_history.history) > 0


def test_history_show():
    new_history = history.History()

    agents = []
    for _ in range(5):
        agents.append(agent.Agent(n_variables=2, n_dimensions=1))

    new_history.dump(agents)

    new_history.show()

    assert True == True


def test_history_save():
    new_history = history.History()

    agents = []
    for _ in range(5):
        agents.append(agent.Agent(n_variables=2, n_dimensions=1))

    new_history.dump(agents)

    new_history.save('history_test.pkl')

    assert os.path.isfile('./history_test.pkl')


def test_history_load():
    new_history = history.History()

    new_history.load('history_test.pkl')

    assert len(new_history.history) > 0

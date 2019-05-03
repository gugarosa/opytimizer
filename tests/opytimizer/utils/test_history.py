import os

import pytest
from opytimizer.core import agent
from opytimizer.utils import history


def test_history_agents():
    new_history = history.History()

    assert type(new_history.agents) == list


def test_history_agents_setter():
    new_history = history.History()

    new_history.agents = [1, 2, 3]

    assert len(new_history.agents) > 0


def test_history_best_agent():
    new_history = history.History()

    assert type(new_history.best_agent) == list


def test_history_best_agent_setter():
    new_history = history.History()

    new_history.best_agent = [1, 2, 3]

    assert len(new_history.best_agent) > 0


def test_history_dump():
    new_history = history.History()

    agents = []
    for _ in range(5):
        agents.append(agent.Agent(n_variables=2, n_dimensions=1))

    new_history.dump(agents, agents[0])

    assert len(new_history.agents) > 0
    assert len(new_history.best_agent) > 0


def test_history_show():
    new_history = history.History()

    agents = []
    for _ in range(5):
        agents.append(agent.Agent(n_variables=2, n_dimensions=1))

    new_history.dump(agents, agents[0])

    new_history.show()

    assert True == True


def test_history_save():
    new_history = history.History()

    agents = []
    for _ in range(5):
        agents.append(agent.Agent(n_variables=2, n_dimensions=1))

    new_history.dump(agents, agents[0])

    new_history.save('models/test.pkl')

    assert os.path.isfile('./models/test.pkl')


def test_history_load():
    new_history = history.History()

    new_history.load('models/test.pkl')

    assert len(new_history.agents) > 0

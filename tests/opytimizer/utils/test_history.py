import numpy as np
import pytest

from opytimizer.core import Agent
from opytimizer.utils.history import History


def make_agents():
    return [Agent(2, 1, [0, 0], [1, 1]) for _ in range(2)]


def test_history_validates_save_agents():
    assert History().save_agents is False

    with pytest.raises(TypeError):
        History("yes")


def test_history_dumps_and_parses_values():
    agents = make_agents()
    history = History(save_agents=True)

    history.dump(
        agents=agents,
        best_agent=agents[0],
        local_position=agents[0].position,
        value=1,
    )
    history.dump(
        agents=agents,
        best_agent=agents[0],
        local_position=agents[0].position,
        value=2,
    )

    agents_pos, agents_fit = history.get_convergence("agents", index=0)
    best_pos, best_fit = history.get_convergence("best_agent")

    assert agents_pos.shape == (2, 2)
    assert agents_fit.shape == (2,)
    assert best_pos.shape == (2, 2)
    assert best_fit.shape == (2,)
    assert history.get_convergence("local_position").shape == (2,)
    assert np.array_equal(history.get_convergence("value"), [1, 2])


def test_history_skips_agents_when_disabled():
    history = History()

    history.dump(agents=make_agents())

    assert not hasattr(history, "agents")

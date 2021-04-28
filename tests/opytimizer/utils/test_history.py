from opytimizer.core import agent
from opytimizer.utils import history


def test_history_save_agents():
    new_history = history.History()

    assert new_history.save_agents is False


def test_history_save_agents_setter():
    new_history = history.History()

    try:
        new_history.save_agents = 'a'
    except:
        new_history.save_agents = True

    assert new_history.save_agents is True


def test_history_dump():
    new_history = history.History(save_agents=True)

    agents = [agent.Agent(n_variables=2, n_dimensions=1, lower_bound=[
                          0, 0], upper_bound=[1, 1]) for _ in range(5)]

    new_history.dump(agents=agents, best_agent=agents[4], value=0)

    assert len(new_history.agents) > 0
    assert len(new_history.best_agent) > 0
    assert new_history.value[0] == 0

    new_history = history.History(save_agents=False)

    new_history.dump(agents=agents)

    assert hasattr(new_history, 'agents') is False


def test_history_get_convergence():
    new_history = history.History(save_agents=True)

    agents = [agent.Agent(n_variables=2, n_dimensions=1, lower_bound=[
                          0, 0], upper_bound=[1, 1]) for _ in range(5)]

    new_history.dump(
        agents=agents, best_agent=agents[4], local_position=agents[0].position, value=0)
    new_history.dump(
        agents=agents, best_agent=agents[4], local_position=agents[0].position, value=0)

    try:
        agents_pos, agents_fit = new_history.get_convergence(
            key='agents', index=5)
    except:
        agents_pos, agents_fit = new_history.get_convergence(
            key='agents', index=0)

    assert agents_pos.shape == (2, 2)
    assert agents_fit.shape == (2,)

    best_agent_pos, best_agent_fit = new_history.get_convergence(
        key='best_agent')

    assert best_agent_pos.shape == (2, 2)
    assert best_agent_fit.shape == (2,)

    try:
        local_position = new_history.get_convergence(
            key='local_position', index=5)
    except:
        local_position = new_history.get_convergence(key='local_position')

    assert local_position.shape == (2,)

    value = new_history.get_convergence(key='value')

    assert value.shape == (2,)

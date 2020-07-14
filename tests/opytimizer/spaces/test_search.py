from opytimizer.spaces import search


def test_search_initialize_agents():
    new_search_space = search.SearchSpace()

    assert new_search_space.agents[0].position[0] != 0


def test_search_clip_limits():
    new_search_space = search.SearchSpace()

    new_search_space.agents[0].position[0] = 20

    new_search_space.clip_limits()

    assert new_search_space.agents[0].position[0] != 20

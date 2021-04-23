from opytimizer.spaces import boolean


def test_boolean_initialize_agents():
    new_boolean_space = boolean.BooleanSpace()

    assert new_boolean_space.agents[0].position[0][0] == 0 or new_boolean_space.agents[0].position[0][0] == 1


def test_boolean_clip_by_bound():
    new_boolean_space = boolean.BooleanSpace()

    new_boolean_space.agents[0].position[0][0] = 20

    new_boolean_space.clip_by_bound()

    assert new_boolean_space.agents[0].position[0][0] == 1

import numpy as np

from opytimizer.optimizers.boolean import umda
from opytimizer.spaces import boolean


def test_umda_params():
    params = {"p_selection": 0.75, "lower_bound": 0.05, "upper_bound": 0.95}

    new_umda = umda.UMDA(params=params)

    assert new_umda.p_selection == 0.75

    assert new_umda.lower_bound == 0.05

    assert new_umda.upper_bound == 0.95


def test_umda_params_setter():
    new_umda = umda.UMDA()

    try:
        new_umda.p_selection = "a"
    except:
        new_umda.p_selection = 0.75

    assert new_umda.p_selection == 0.75

    try:
        new_umda.p_selection = -1
    except:
        new_umda.p_selection = 0.75

    assert new_umda.p_selection == 0.75

    try:
        new_umda.lower_bound = "a"
    except:
        new_umda.lower_bound = 0.05

    assert new_umda.lower_bound == 0.05

    try:
        new_umda.lower_bound = -1
    except:
        new_umda.lower_bound = 0.05

    assert new_umda.lower_bound == 0.05

    try:
        new_umda.upper_bound = "a"
    except:
        new_umda.upper_bound = 0.95

    assert new_umda.upper_bound == 0.95

    try:
        new_umda.upper_bound = -1
    except:
        new_umda.upper_bound = 0.95

    assert new_umda.upper_bound == 0.95

    try:
        new_umda.upper_bound = 0.04
    except:
        new_umda.upper_bound = 0.95

    assert new_umda.upper_bound == 0.95


def test_umda_calculate_probability():
    new_umda = umda.UMDA()

    boolean_space = boolean.BooleanSpace(n_agents=5, n_variables=2)

    probs = new_umda._calculate_probability(boolean_space.agents)

    assert probs.shape == (2, 1)


def test_umda_sample_position():
    new_umda = umda.UMDA()

    probs = np.zeros((1, 1))

    position = new_umda._sample_position(probs)

    assert position == 1


def test_umda_update():
    new_umda = umda.UMDA()

    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_umda.update(boolean_space)

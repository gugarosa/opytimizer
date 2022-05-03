import numpy as np

from opytimizer.optimizers.social import isa
from opytimizer.spaces import search
from opytimizer.utils import constant


def test_isa_params():
    params = {"w": 0.7, "tau": 0.3}

    new_isa = isa.ISA(params=params)

    assert new_isa.w == 0.7

    assert new_isa.tau == 0.3


def test_isa_params_setter():
    new_isa = isa.ISA()

    try:
        new_isa.w = "a"
    except:
        new_isa.w = 0.7

    try:
        new_isa.w = -1
    except:
        new_isa.w = 0.7

    assert new_isa.w == 0.7

    try:
        new_isa.tau = "b"
    except:
        new_isa.tau = 0.3

    try:
        new_isa.tau = -1
    except:
        new_isa.tau = 0.3

    assert new_isa.tau == 0.3


def test_isa_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_isa = isa.ISA()
    new_isa.compile(search_space)

    try:
        new_isa.local_position = 1
    except:
        new_isa.local_position = np.array([1])

    assert new_isa.local_position == 1

    try:
        new_isa.velocity = 1
    except:
        new_isa.velocity = np.array([1])

    assert new_isa.velocity == 1


def test_isa_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_isa = isa.ISA()
    new_isa.compile(search_space)

    new_isa.evaluate(search_space, square)

    assert search_space.best_agent.fit != constant.FLOAT_MAX


def test_isa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_isa = isa.ISA()
    new_isa.compile(search_space)

    for _ in range(10):
        new_isa.update(search_space, square)

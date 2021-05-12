import numpy as np

from opytimizer.optimizers.social import qsa
from opytimizer.spaces import search
from opytimizer.utils import constant


def test_qsa_calculate_queue():
    new_qsa = qsa.QSA()

    q_1, q_2, q_3 = new_qsa._calculate_queue(10, 1, 1, 1)

    assert q_1 == 3
    assert q_2 == 3
    assert q_3 == 3

    q_1, q_2, q_3 = new_qsa._calculate_queue(10, constant.EPSILON - 0.1, 1, 1)

    assert q_1 == 3
    assert q_2 == 3
    assert q_3 == 3


def test_qsa_business_one():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_qsa = qsa.QSA()

    new_qsa._business_one(search_space.agents, square, 0.1)
    new_qsa._business_one(search_space.agents, square, 100)


def test_qsa_business_two():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_qsa = qsa.QSA()

    new_qsa._business_two(search_space.agents, square)


def test_qsa_business_three():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_qsa = qsa.QSA()

    new_qsa._business_three(search_space.agents, square)


def test_qsa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=100, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_qsa = qsa.QSA()

    new_qsa.update(search_space, square, 1, 10)

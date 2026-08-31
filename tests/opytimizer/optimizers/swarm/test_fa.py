from opytimizer.optimizers.swarm import fa
from opytimizer.spaces import search


def test_fa_params():
    params = {"alpha": 0.5, "beta": 0.2, "gamma": 1.0}

    new_fa = fa.FA(params=params)

    assert new_fa.alpha == 0.5

    assert new_fa.beta == 0.2

    assert new_fa.gamma == 1.0


def test_fa_update():
    new_fa = fa.FA()

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_fa.update(search_space, 100)

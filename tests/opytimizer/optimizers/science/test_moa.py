# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import pytest

from opytimizer.optimizers.science import moa
from opytimizer.spaces import search


def test_moa_params():
    params = {
        "alpha": 1.0,
        "rho": 2.0,
    }

    new_moa = moa.MOA(params=params)

    assert new_moa.alpha == 1.0

    assert new_moa.rho == 2.0


def test_moa_compile_rejects_non_square_population():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])
    new_moa = moa.MOA()

    with pytest.raises(ValueError, match=r"^`n_agents` must be a perfect square\.$"):
        new_moa.compile(search_space)


def test_moa_compile_accepts_square_population():
    search_space = search.SearchSpace(n_agents=9, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])
    new_moa = moa.MOA()

    new_moa.compile(search_space)


def test_moa_update():
    search_space = search.SearchSpace(n_agents=9, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_moa = moa.MOA()
    new_moa.compile(search_space)

    new_moa.update(search_space)

import numpy as np

from opytimizer.utils import decorator


def test_hyper_spanning():
    lb = np.full(1, 10)
    ub = np.full(1, 20)

    @decorator.hyper_spanning(lb, ub)
    def call(x):
        return np.sum(x)

    y = call(np.array([[0.5], [0.5]]))

    assert y == 30


def test_pre_evaluation():
    @decorator.pre_evaluation
    def call(obj, x):
        return x

    def f():
        return True

    assert f() == True

    call(f, 1, hook=None)

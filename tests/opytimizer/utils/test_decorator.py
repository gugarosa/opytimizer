import pytest

from opytimizer.utils import decorator


def test_pre_evaluation():
    @decorator.pre_evaluation
    def call(obj, x):
        return x

    def f():
        return True

    assert f() == True

    call(f, 1, hook=None)

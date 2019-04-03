import numpy as np
import pytest
from opytimizer.functions import multi

def test_multi():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    try:
        new_multi = multi.Multi()
    except:
        new_multi = multi.Multi(functions=[square, cube], weights=[0.5, 0.5], method='weight_sum')


    assert new_multi.functions[0].pointer(2) == 4

    assert new_multi.functions[1].pointer(2) == 8

    assert new_multi.method == 'weight_sum'

    assert new_multi.pointer(2) == 6
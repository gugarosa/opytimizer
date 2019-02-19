import pytest

from opytimizer.functions import internal

def test_function_internal_build():

    def square(x):
        return x**2
    
    new_internal = internal.Internal(pointer=square)

    assert new_internal.built == True

def test_function_internal_type():

    def square(x):
        return x**2
    
    new_internal = internal.Internal(pointer=square)

    assert new_internal.type == 'internal'
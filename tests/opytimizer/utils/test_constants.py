import sys

import pytest

from opytimizer.utils import constants


def test_constants():
    assert constants.EPSILON == 10e-10

    assert constants.FLOAT_MAX == sys.float_info.max

    assert constants.HISTORY_KEYS == ['agents', 'best_agent', 'local']

    assert constants.N_ARGS_FUNCTION == {
        'SUM': 2,
        'SUB': 2,
        'MUL': 2,
        'DIV': 2,
        'EXP': 1,
        'SQRT': 1,
        'LOG': 1,
        'ABS': 1,
        'SIN': 1,
        'COS': 1
    }

    assert constants.TEST_EPSILON == 5

    assert constants.TOURNAMENT_SIZE == 2

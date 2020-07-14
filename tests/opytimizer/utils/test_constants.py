import sys

from opytimizer.utils import constants


def test_constants():
    assert constants.EPSILON == 1e-32

    assert constants.FLOAT_MAX == sys.float_info.max

    assert constants.HISTORY_KEYS == ['agents', 'best_agent', 'local']

    assert constants.LIGHT_SPEED == 3e5

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

    assert constants.TEST_EPSILON == 100

    assert constants.TOURNAMENT_SIZE == 2
